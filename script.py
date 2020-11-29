
import numpy as np
import torch.nn as nn
import pandas as pd
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from visdom import Visdom
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from copy import copy


def main():
    train()


def test():
    train_file = '69225_Inputs_Yes_at_end_FlightTrainNew1.csv'
    test_file = '29668_Inputs_FlightTestNoYNew1.csv'
    outfile = 'answers.csv'
    answer_list = []
    train_df_numpy, test_df_numpy, labels_numpy = convertTrainTestToNumpy(train_file, test_file)
    input_dim = 303
    output_dim = 1
    model = Network(input_dim, output_dim)
    model.load_state_dict(torch.load('test_best_weights.pth'))
    model = model.cuda()
    model.eval()
    for i, (data_point, label_point) in enumerate(zip(train_df_numpy, labels_numpy), 1):
        torch_data_point = torch.from_numpy(data_point).unsqueeze(0)
        torch_data_point = torch_data_point.float().cuda()
        answer = model(torch_data_point)
        answer = answer.round().long().item()
        answer = "YES" if answer == 1 else "NO"
        assert answer == 'YES'
        answer = f"{i},{answer}"
        answer_list.append(answer)
    answer_list.insert(0, "Ob,Cancelled")
    csv_payload = '\n'.join(answer_list)

    with open(outfile, 'w+') as f:
        print(csv_payload, file=f)


def train():
    tensor_file = 'runs/final_full_2layer_pass_status_sched_decay'
    save_weights = 'decay_sched_pass_status_final_full_hidden_best_weights.pth'
    train_split = 0.9
    model_class = Network

    train_file = '69225_Inputs_Yes_at_end_FlightTrainNew1.csv'
    test_file = '29668_Inputs_FlightTestNoYNew1.csv'
    train_df_numpy, test_df_numpy, labels_numpy = convertTrainTestToNumpy(train_file, test_file)
    input(f"Shape of train: {train_df_numpy.shape}, test: {test_df_numpy.shape}")
    # HYPER PARAMETERS
    epochs = 100000
    lr_rate = 1e-3
    input_dim = 324  # 272 good?
    output_dim = 1
    # END HYPER PARAMETERS
    # MODEL and LOSS
    model = model_class(input_dim, output_dim)
    model = model.cuda()
    criterion = nn.BCELoss()  # computes softmax and then the cross entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate, weight_decay=1e-4)
    # END MODEL and LOSS
    # DATASET CREATION and TRAIN AND VALID LOADER
    dataset = Dataset(train_df_numpy, labels_numpy)
    dataset_size = len(dataset)
    train_dataset_size = int(train_split * dataset_size)
    batch_size = train_dataset_size
    valid_dataset_size = dataset_size - train_dataset_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_dataset_size, valid_dataset_size])
    train_loader = data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False)
    if train_split < 1:
        valid_loader = data.DataLoader(dataset=val_set, batch_size=valid_dataset_size, shuffle=False)
    # DATASET CREATION and TRAIN AND VALID LOADER END
    writer = SummaryWriter(tensor_file)
    best_loss = float('inf')
    best_acc = 0
    for epoch in range(epochs):
        accuracy_total_train = 0
        loss_total = 0
        for i, (features, labels) in enumerate(train_loader):  # iter = 0
            features = features.float().cuda()
            labels = labels.float().cuda()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                model.eval()
                accuracy = calc_accuracy(outputs.data, labels)
                model.train()
                accuracy_total_train += accuracy
                loss_total += loss.item()
        if train_split < 1:
            with torch.no_grad():
                model.eval()
                accuracy_total_valid = 0
                for i_, (features_, labels_) in enumerate(valid_loader):
                    features_ = features_.float().cuda()
                    labels_ = labels_.float().cuda()
                    outputs_ = model(features_)
                    accuracy_ = calc_accuracy(outputs_.data, labels_)
                    accuracy_total_valid += accuracy_
                n_of_iter = i + 1
                n_of_iter_ = i_ + 1
                avg_loss = loss_total / n_of_iter
                avg_acc = accuracy_total_train / n_of_iter
                avg_acc_ = accuracy_total_valid / n_of_iter_
                print(f"Epoch: {epoch}. Loss: {avg_loss}. Accuracy: {avg_acc}. Valid Accuracy: {avg_acc_}")
                if avg_loss < best_loss:
                    best_loss = avg_loss
                if avg_acc_ > best_acc:
                    best_acc = avg_acc_
                    savepath = save_weights
                    torch.save(model.state_dict(), savepath)
                writer.add_scalar("Loss/train", avg_loss, epoch)
                writer.add_scalar("acc/valid", avg_acc_, epoch)
                writer.flush()
                model.train()
        else:
            with torch.no_grad():
                model.eval()
                n_of_iter = i + 1
                avg_acc = accuracy_total_train / n_of_iter
                avg_loss = loss_total / n_of_iter
                print(f"Epoch: {epoch}. Loss: {avg_loss}. Accuracy: {avg_acc}.")
                if avg_loss < best_loss:
                    best_loss = avg_loss
                if avg_acc > best_acc:
                    best_acc = avg_acc
                    savepath = save_weights
                    torch.save(model.state_dict(), savepath)
                writer.add_scalar("Loss/train", avg_loss, epoch)
                writer.add_scalar("acc/train", avg_acc, epoch)
                writer.flush()
                model.train()


def convertTrainTestToNumpy(train_file, test_file):

    raw_test_df = pd.read_csv(test_file)
    raw_train_df = pd.read_csv(train_file)

    train_features, label_name, raw_train_df = __createFeaturesForPd(raw_train_df)

    test_features, label_name_, raw_test_df = __createFeaturesForPd(raw_test_df)

    w_delay_train = []
    for w_delay in raw_train_df['WEATHER_DELAY']:
        if w_delay != w_delay:
            w_delay_train.append(0)
        else:
            w_delay_train.append(1)

    w_delay_test = []
    for w_delay in raw_test_df['WEATHER_DELAY']:
        if w_delay != w_delay:
            w_delay_test.append(0)
        else:
            w_delay_test.append(1)

    raw_test_df['WEATHER_DELAY'] = w_delay_test
    raw_train_df['WEATHER_DELAY'] = w_delay_train

    train_df = raw_train_df[train_features]
    test_df = raw_test_df[test_features]

    feature_to_add_to_train = __getListOfMissingFeatures(test_features, train_features)
    feature_to_add_to_test = __getListOfMissingFeatures(train_features, test_features)
    train_df = __addMissingCategories(feature_to_add_to_train, train_df)
    test_df = __addMissingCategories(feature_to_add_to_test, test_df)
    train_df = train_df.reindex(sorted(train_df.columns), axis=1)
    test_df = test_df.reindex(sorted(test_df.columns), axis=1)

    labels_df = raw_train_df[label_name]

    train_df_numpy = train_df.to_numpy()
    test_df_numpy = test_df.to_numpy()
    labels_numpy = labels_df.to_numpy().astype(float)

    for data_point in train_df_numpy:
        data_point[train_df.columns.get_loc("Passengers")] = data_point[train_df.columns.get_loc(
            "Passengers")] / 61122
        data_point[train_df.columns.get_loc("Flights")] = data_point[train_df.columns.get_loc(
            "Flights")] / 788
        data_point[train_df.columns.get_loc("Distance")] = (data_point[train_df.columns.get_loc(
            "Distance")] - 67) / (4983 - 67)
        data_point[train_df.columns.get_loc("Rank")] = (data_point[
                                                                     train_df.columns.get_loc("Rank")] - 1) / (
                                                                            35 - 1)
        data_point[train_df.columns.get_loc("Average.Passengers")] = (data_point[
                                                            train_df.columns.get_loc("Average.Passengers")] - 6.003824e+06) / (
                                                               4.694981e+07 - 6.003824e+06)

    for data_point in test_df_numpy:
        data_point[test_df.columns.get_loc("Passengers")] = data_point[test_df.columns.get_loc(
            "Passengers")] / 61122
        data_point[test_df.columns.get_loc("Flights")] = data_point[test_df.columns.get_loc(
            "Flights")] / 788
        data_point[test_df.columns.get_loc("Distance")] = (data_point[test_df.columns.get_loc(
            "Distance")] - 67) / (4983 - 67)
        data_point[test_df.columns.get_loc("Rank")] = (data_point[
                                                                     test_df.columns.get_loc("Rank")] - 1) / (
                                                                            35 - 1)
        data_point[train_df.columns.get_loc("Average.Passengers")] = (data_point[
                                                                          train_df.columns.get_loc(
                                                                              "Average.Passengers")] - 6.003824e+06) / (
                                                                             4.694981e+07 - 6.003824e+06)

    return train_df_numpy, test_df_numpy, labels_numpy


def __addMissingCategories(feature_to_add_to_train, train_features_df):
    train_row_length = train_features_df.shape[0]
    twoDZeros = np.zeros((train_row_length, len(feature_to_add_to_train)))
    newFeatureDataFrame = pd.DataFrame(twoDZeros, columns=feature_to_add_to_train)
    train_features_df = pd.concat([train_features_df, newFeatureDataFrame], axis=1)
    return train_features_df


def __getListOfMissingFeatures(test_features, train_features):
    set_train_feature = set(train_features)
    feature_to_add_to_train = set()
    for test_feature in test_features:
        if test_feature not in set_train_feature:
            feature_to_add_to_train.add(test_feature)
    return list(feature_to_add_to_train)


def __createFeaturesForPd(raw_df):
    airlineArrLabels, raw_df = __addCategory('AIRLINE', 'c', raw_df)
    originLabels, raw_df = __addCategory('Origin_airport', 'a', raw_df)
    destinationLabels, raw_df = __addCategory('Destination_airport', 'b', raw_df)
    weekDayLabels, raw_df = __addCategory('DAY_OF_WEEK', 'dayofweek', raw_df)
    rankStatusLabels, raw_df = __addCategory('Rank.Status', 'rankStatus', raw_df)
    monthLabels, raw_df = __addCategory('MONTH', 'month_', raw_df)
    dayLabels, raw_df = __addCategory('DAY', 'day_of_month', raw_df)
    raw_df = raw_df.replace(to_replace="YES", value=1)
    raw_df = raw_df.replace(to_replace="NO", value=0)
    label_name = ['Cancelled']
    feature_names = ['Passengers', 'Flights', 'Distance', 'Rank', 'DIVERTED', 'WEATHER_DELAY']
    __addLabelsToFeatures(['Average.Passengers'], feature_names)
    timeMap = {}
    for i in range(2400):
        if i < 400:
            timeMap[i] = 1
        elif i < 800:
            timeMap[i] = 2
        elif i < 1200:
            timeMap[i] = 3
        elif i < 1600:
            timeMap[i] = 4
        elif i < 2000:
            timeMap[i] = 5
        else:
            timeMap[i] = 6
        assert i <= 2400
    t1_list = raw_df['SCHEDULED_DEPARTURE'].tolist()
    t2_list = raw_df['SCHEDULED_TIME'].tolist()
    t3_list = raw_df['SCHEDULED_ARRIVAL'].tolist()
    assert len(t1_list) == len(t2_list) == len(t3_list)
    for i in range(len(t1_list)):
        t1_list[i] = timeMap[t1_list[i]]
        t2_list[i] = timeMap[t2_list[i]]
        t3_list[i] = timeMap[t3_list[i]]
    raw_df['DEPART_'] = t1_list
    raw_df['TIME_'] = t2_list
    raw_df['ARRIVE_'] = t3_list
    departLabels, raw_df = __addCategory('DEPART_', 'depart_time', raw_df)
    timeLabels, raw_df = __addCategory('TIME_', 'time_time', raw_df)
    arriveLabels, raw_df = __addCategory('ARRIVE_', 'arrive_time', raw_df)

    __addLabelsToFeatures(airlineArrLabels, feature_names)
    __addLabelsToFeatures(originLabels, feature_names)
    __addLabelsToFeatures(destinationLabels, feature_names)
    __addLabelsToFeatures(monthLabels, feature_names)
    __addLabelsToFeatures(dayLabels, feature_names)
    __addLabelsToFeatures(rankStatusLabels, feature_names)
    __addLabelsToFeatures(departLabels, feature_names)
    __addLabelsToFeatures(timeLabels, feature_names)
    __addLabelsToFeatures(arriveLabels, feature_names)
    return feature_names, label_name, raw_df


def __fixPassTrafficNAValues(train_features_df):
    train_features_df['Pass.Traffic'].fillna((train_features_df['Pass.Traffic'].mean()), inplace=True)


def __fixAirlineDelayValues(train_features_df):
    train_features_df['AIR_SYSTEM_DELAY'].fillna((train_features_df['AIR_SYSTEM_DELAY'].mean()), inplace=True)
    train_features_df['SECURITY_DELAY'].fillna((train_features_df['SECURITY_DELAY'].mean()), inplace=True)
    train_features_df['AIRLINE_DELAY'].fillna((train_features_df['AIRLINE_DELAY'].mean()), inplace=True)
    train_features_df['LATE_AIRCRAFT_DELAY'].fillna((train_features_df['LATE_AIRCRAFT_DELAY'].mean()), inplace=True)
    train_features_df['WEATHER_DELAY'].fillna((train_features_df['WEATHER_DELAY'].mean()), inplace=True)


def __addLabelsToFeatures(airlineArrLabels, feature_names):
    feature_names.extend(airlineArrLabels)


def __addCategory(featureToCategorize, suffix, train_df):
    labelEncoder = LabelEncoder()
    rows_of_feature = train_df[featureToCategorize]
    airline = labelEncoder.fit_transform(rows_of_feature)
    train_df[f'{featureToCategorize}_Numerical'] = airline
    oneHotEncoder = OneHotEncoder()
    airlineArr = oneHotEncoder.fit_transform(train_df[[f'{featureToCategorize}_Numerical']]).toarray()
    airlineArrLabels = [str(cls_label) + suffix for cls_label in labelEncoder.classes_]
    airlineFeatures = pd.DataFrame(airlineArr, columns=airlineArrLabels)
    train_df = pd.concat([train_df, airlineFeatures], axis=1)
    return airlineArrLabels, train_df

class Dataset(data.Dataset):

    def __init__(self, __data_points, __data_labels):
        self.__data_points = __data_points
        self.__data_labels = __data_labels

    def __getitem__(self, index):
        return torch.from_numpy(self.__data_points[index]), torch.from_numpy(self.__data_labels[index])

    def __len__(self):
        return len(self.__data_points)


class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        output = self.sigmoid(x)
        return output

class Network(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim//2)
        self.b1 = nn.BatchNorm1d(input_dim//2)
        self.fc2 = nn.Linear(input_dim//2, input_dim//2//2)
        self.b2 = nn.BatchNorm1d(input_dim//2//2)
        self.fc3 = nn.Linear(input_dim//2//2, input_dim//2//2//2)
        self.b3 = nn.BatchNorm1d(input_dim//2//2//2)
        self.fc4 = nn.Linear(input_dim//2//2//2, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.b1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.b2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.b3(x)
        x = self.relu(x)
        x = self.fc4(x)
        output = self.sigmoid(x)
        return output

class Network3Hidden(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Network3Hidden, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim // 2)
        self.b1 = nn.BatchNorm1d(input_dim // 2)
        self.fc2 = nn.Linear(input_dim // 2, input_dim // 2 // 2)
        self.b2 = nn.BatchNorm1d(input_dim // 2 // 2)
        self.fc3 = nn.Linear(input_dim // 2 // 2, input_dim // 2 // 2 // 2)
        self.b3 = nn.BatchNorm1d(input_dim // 2 // 2 // 2)
        self.fc4 = nn.Linear(input_dim // 2 // 2 // 2, input_dim // 2 // 2 // 2 // 2)
        self.b4 = nn.BatchNorm1d(input_dim // 2 // 2 // 2 // 2)
        self.fc5 = nn.Linear(input_dim // 2 // 2 // 2 // 2, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.b1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.b2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.b3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.b4(x)
        x = self.relu(x)
        x = self.fc5(x)
        output = self.sigmoid(x)
        return output


def calc_accuracy(Y_hat, Y):
    Y = Y.long()
    Y_hat_rounded = Y_hat.round().long()
    max_vals, max_indices = Y_hat.max(1)
    max_indices = max_indices.unsqueeze(1)

    n = max_indices.size(0)  # index 0 for extracting the # of elements
    num_corrected = (Y_hat_rounded == Y).sum()

    acc = num_corrected.item() / n
    try:
        assert acc <= 1
    except:
        raise ValueError(acc)
    return acc


if __name__ == '__main__':
    main()