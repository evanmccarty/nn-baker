import numpy as np
import operator
import keras
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

#unet architecture

def conv_block(x, num_filters):
    x = Conv2D(num_filters, (3, 3), padding="same")(x)
    x = Dropout(0.3)(x)
    x = Activation("relu")(x)
    x = Conv2D(num_filters, (3, 3), padding="same")(x)
    x = Dropout(0.3)(x)
    x = Activation("relu")(x)
    return x

def build_unet(size = 128, nodes_per_layer = [128,256,512,1024]):
    input_size = Input((size, size, 1))
    conv_list = []
    x = input_size

    #Assemble the encoded convolutional blocks
    for nodes in nodes_per_layer:
        x = conv_block(x, nodes)
        conv_list.append(x)
        x = MaxPool2D((2, 2))(x)

    # Last layer of unet
    x = conv_block(x, nodes_per_layer[-1])

    nodes_per_layer.reverse()
    conv_list.reverse()
    ## Decode the convolutional blocks
    for i, f in enumerate(nodes_per_layer):
        up_sampling = UpSampling2D((2, 2))(x)
        matching_conv = conv_list[i]
        x = Concatenate()([up_sampling, matching_conv])
        x = conv_block(x, f)

    ## Output
    x = Conv2D(1, (1, 1), padding="same")(x)
    x = Activation("sigmoid")(x)
    return Model(input_size, x)

def compute_distance_square(pointa, pointb):
    pointa[0] = round(pointa[0], 2)
    pointa[1] = round(pointa[1], 2)
    pointb[0] = round(pointb[0], 2)
    pointb[1] = round(pointb[1], 2)
    return math.sqrt((pointa[0] - pointb[0]) ** 2 + (pointa[1] - pointb[1]) ** 2)

#Training

m = 25.6
array_size = 256.0

def encode(vertices, array_size = 100.0, m = 10.0):
    encoding = np.zeros((int(array_size), int(array_size)))
    bin_size = m/array_size
    for vertex in vertices:
        x_val = vertex[0]
        y_val = vertex[1]
        x_bin = int(x_val/bin_size)
        y_bin = int(y_val/bin_size)
        encoding[x_bin][y_bin]+=1
    return encoding

def encode_sols(sols, vertices, array_size = 100.0, m = 10.0):
    encoding = np.zeros((int(array_size), int(array_size)))
    bin_size = m / array_size
    for sol in sols:
        x_val = vertices[sol][0]
        y_val = vertices[sol][1]
        x_bin = int(x_val/bin_size)
        y_bin = int(y_val/bin_size)
        encoding[x_bin][y_bin]+=1
    return encoding

def decode(vertices, array):
    pred_to_return = []
    #print(len(array[0]))
    bin_size = m/array_size
    for vertex in vertices:
        x_val = vertex[0]
        y_val = vertex[1]
        x_bin = int(x_val/bin_size)
        y_bin = int(y_val/bin_size)
        pred_to_return.append(array[0][x_bin][y_bin])
    return pred_to_return

a_s = 1280
m_s = a_s*0.1


np.random.seed(seed=8675309)
file_numbers = [5,6,7]


data_to_train = []
data_to_train_sols = []
number = 0
for file_num in file_numbers:
    file_to_import = "./data.npy"
    to_add = np.load(file_to_import, allow_pickle=True)

    for case in to_add:
        number+=1
        print(number)
        vertices = case[4]
        edges = {}
        for ind in range(len(vertices) + 1):
            edges[ind] = []
        for pair in case[5]:
            edges[pair[0]].append(pair[1])
        region = np.array(encode(vertices,array_size=a_s,m=m_s))
        solutions = case[7]
        sol_region = np.array(encode_sols(solutions,vertices,array_size=a_s,m=m_s))
        for x_axis in range(0,a_s,128):
            #print(x_axis)
            for y_axis in range(0,a_s,128):
                sm_region = np.zeros((128,128))
                sm_sol = np.zeros((128,128))
                for x_small in range(128):
                    for y_small in range(128):
                        sm_region[x_small][y_small] += region[x_axis+x_small][y_axis+y_small]
                        sm_sol[x_small][y_small] += sol_region[x_axis + x_small][y_axis + y_small]

                data_to_train.append(sm_region)
                data_to_train.append(np.flip(sm_region, 0))
                data_to_train.append(np.flip(sm_region, 1))
                data_to_train.append(np.flip(sm_region, (0, 1)))
                data_to_train_sols.append(sm_sol)
                data_to_train_sols.append(np.flip(sm_sol, 0))
                data_to_train_sols.append(np.flip(sm_sol, 1))
                data_to_train_sols.append(np.flip(sm_sol, (0, 1)))

trlv = np.array(data_to_train)
trlvs = np.array(data_to_train_sols)
trlv = np.expand_dims(trlv, -1)

model = build_unet()
model.compile(optimizer='adam', loss="binary_crossentropy",
              metrics=['accuracy',
                       tf.keras.metrics.BinaryCrossentropy(),
                       tf.keras.metrics.RootMeanSquaredError(),
                       tf.keras.metrics.AUC()])
print(model.summary(100))
#model = keras.models.load_model('./model.h5')
model.fit(trlv[100:], trlvs[100:], epochs=3, batch_size=100,
                      validation_data=(trlv[:100], trlvs[:100]), shuffle=True)

model.save('./model.h5')

##testing code


m = 25.6
array_size = 256.0



a_s = 2560
m_s = a_s*0.1

np.random.seed(seed=8675309)

model = keras.models.load_model('./model.h5')

print(model.summary(100))

def box_unit_ball(vertex_index_pair,vertices):
    a = vertex_index_pair[0]
    b = vertex_index_pair[1]
    return compute_distance_square(vertices[a], vertices[b]) <= 1


for file_num in file_numbers:
    #file_to_import = "./venv/baker_gaussian/mis_2Ddata_gaussian128.0_"+str(file_num)+"_approx.npy"
    file_to_import = "./data.npy"
    to_add = np.load(file_to_import, allow_pickle=True)

    for case in to_add:
        dec_graph = []
        vertices = case[4]
        edges = {}
        for ind in range(len(vertices) + 1):
            edges[ind] = []
        for pair in case[5]:
            edges[pair[0]].append(pair[1])
        region = np.array(encode(vertices,array_size=a_s,m=m_s))
        solutions = case[7]
        pred_map = np.zeros((1,a_s,a_s))
        for x_axis in range(0,a_s,128):
            #print(x_axis)
            for y_axis in range(0,a_s,128):
                sm_region = np.zeros((128,128))
                for x_small in range(128):
                    for y_small in range(128):
                        sm_region[x_small][y_small] = region[x_axis+x_small][y_axis+y_small]
                to_model = sm_region
                to_model = [to_model]
                to_model = np.expand_dims(to_model,-1)
                prediction_map = model.predict(to_model)
                for x_small in range(128):
                    for y_small in range(128):
                        pred_map[0][x_axis+x_small][y_axis+y_small] = prediction_map[0][x_small][y_small]
        predictions = decode(vertices, pred_map)
        sorted_v_pred_pairs = sorted(enumerate(predictions), key=operator.itemgetter(1), reverse=True)
        sorted_v_indices = [ind for ind, elem in sorted_v_pred_pairs]
        dec_graph = []
        for ind in sorted_v_indices:
            needed = True
            for adj in edges[ind]:
                if (adj in dec_graph):
                    needed = False
                    break
            if (needed):
                dec_graph.append(ind)
        print(file_num,len(dec_graph),len(solutions))

