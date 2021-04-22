import keras
import keras.backend as K
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger, TensorBoard
import os
import sys
import pandas as pd
from sklearn.model_selection import KFold

from utils import get_score_from_all_slices
from model import create_xception_unet_n
from loss import get_loss, dice
from data_brainmask import create_train_date_generator, create_val_date_generator


#default from github source
#data_file_path = '/data/data/ATLAS_h5/ATLAS.h5'
#if on CHPC1
if os.path.isdir("/scratch/hasm"):
    print("".join(["ERROR: data does not exist on CHPCv2"]))
    # scratch_dir = os.path.join("/scratch/hasm")
    sys.exit()
    # print("".join(["ON old CHPC (~2020): using scratch directory (", str(scratch_dir), ")"]))
    # data_file_path = os.path.join(scratch_dir, "Data", "Lesion", "ATLAS_R1.1", "train.h5")
#if on CHPC3
elif os.path.isdir("/scratch/sungminha"):
    scratch_dir = os.path.join("/scratch/sungminha")
    print("".join(["ON new CHPC (2021~): using scratch directory (", str(scratch_dir), ")"]))

    #for full data
    data_file_path = os.path.join(scratch_dir, "Data", "Lesion", "ATLAS_R1.1_Protocols", "0_freesurfer-6.0.0_full_improved_symlink", "train_brainmask.h5") 

    #for subset
    # data_file_path = os.path.join(scratch_dir, "Data", "Lesion", "ATLAS_R1.1_Protocols", "0_freesurfer-6.0.0_full_improved_symlink", "train_brainmask.h5") 

print("".join(["data_file_path: (", str(data_file_path),")"]))
    
if not (os.path.isfile(data_file_path)):
    print("".join(["ERROR: data_file_path (", str(data_file_path), ") does not exist."]))
    sys.exit()
pretrained_weights_file = None
input_shape = (247, 251, 1) #for FS space data
# input_shape = (256, 256, 1) #for FS space data
dim_z = 256
#14336, 256, 256
batch_size = 8
num_folds = 5
num_epoch = 100
# num_subjects = 56 #for subset
num_subjects = 229 # for full

print("".join(["data_file_path: (", data_file_path, ")"]), flush=True)
print("".join(["batch_size: (", str(batch_size), ")"]), flush=True)
print("".join(["num_folds: (", str(num_folds), ")"]), flush=True)
print("".join(["dim_z: (", str(dim_z), ")"]), flush=True)
print("".join(["num_epoch: (", str(num_epoch), ")"]), flush=True)
print("".join(["num_subjects: (", str(num_subjects), ")"]), flush=True)
print("".join(["input_shape:"]), flush=True)
print(input_shape, flush=True)

def train(fold, train_patient_indexes, val_patient_indexes):

    log_dir = 'fold_' + str(fold) + '/'
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    num_slices_train = len(train_patient_indexes) * 189
    num_slices_val = len(val_patient_indexes) * 189

    # Create model
    K.clear_session()
    model = create_xception_unet_n(input_shape=input_shape, pretrained_weights_file=pretrained_weights_file)
    model.compile(optimizer=Adam(lr=1e-3), loss=get_loss, metrics=[dice])

    # Get callbacks
    checkpoint = ModelCheckpoint(log_dir + 'ep={epoch:03d}-loss={loss:.3f}-val_loss={val_loss:.3f}.h5', verbose=1,
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, min_delta=1e-3, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1)
    csv_logger = CSVLogger(log_dir + 'record.csv')
    tensorboard = TensorBoard(log_dir=log_dir)

    # train the model
    model.fit_generator(
        create_train_date_generator(patient_indexes=train_patient_indexes, h5_file_path=data_file_path, batch_size=batch_size),
        steps_per_epoch=max(1, num_slices_train // batch_size),
        validation_data=create_val_date_generator(patient_indexes=val_patient_indexes, h5_file_path=data_file_path, batch_size=9),
        validation_steps=max(1, num_slices_val // 9),
        epochs=num_epoch,
        initial_epoch=0,
        callbacks=[checkpoint, reduce_lr, early_stopping, tensorboard, csv_logger])
    model.save_weights(log_dir + 'trained_final_weights.h5')
    #save model itself
    model.save(os.path.join(log_dir, 'trained_final_model'))
    # model.save(os.path.join(log_dir, 'trained_final_model.h5'))


    # Evaluate model
    predicts = []
    labels = []
    f = create_val_date_generator(patient_indexes=val_patient_indexes, h5_file_path=data_file_path)
    for _ in range(num_slices_val):
        img, label = f.__next__()
        predicts.append(model.predict(img))
        labels.append(label)
    predicts = np.array(predicts)
    labels = np.array(labels)
    score_record = get_score_from_all_slices(labels=labels, predicts=predicts)

    # save score
    df = pd.DataFrame(score_record)
    df.to_csv(os.path.join(log_dir, 'score_record.csv'), index=False)

    # print score
    mean_score = {}
    for key in score_record.keys():
        print('In fold ', fold, ', average', key, ' value is: \t ', np.mean(score_record[key]))
        mean_score[key] = np.mean(score_record[key])

    # exit training
    K.clear_session()
    return mean_score


def main():
    # prepare indexes of patients for training and validation, respectively
    num_patients = num_subjects
    patients_indexes = np.array([i for i in range(num_patients)])
    kf = KFold(n_splits=num_folds, shuffle=False)

    # train, and record the scores of each fold
    folds_score = []
    for fold, (train_patient_indexes, val_patient_indexes) in enumerate(kf.split(patients_indexes)):
        fold_mean_score = train(fold=fold, train_patient_indexes=train_patient_indexes, val_patient_indexes=val_patient_indexes)
        folds_score.append(fold_mean_score)

    # calculate average score
    print('Final score from ', num_folds, ' folds cross validation:')
    final_score = {}
    for key in folds_score[0].keys():
        scores = []
        for i in range(num_folds):
            scores.append(folds_score[i][key])
        final_score[key] = np.mean(scores)
        print(key, ' score: \t', final_score[key])

    # save final score
    df = pd.DataFrame(final_score, index=[0])
    df.to_csv('x_net_final_score.csv', index=False)


if __name__ == '__main__':
    import tensorflow as tf
    print("tf.config.experimental.list_physical_devices('GPU')")
    print(tf.config.experimental.list_physical_devices('GPU'))
    del tf

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()

