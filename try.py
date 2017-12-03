import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.utils import shuffle
import xgboost as xgb
from sklearn.model_selection import train_test_split
# from keras.models import Input, Model
# from keras.layers import Dense
# from keras import optimizers


def get_wmae(y, y_pred):
    tmp = [abs(y[i] - y_pred[i]) for i in range(len(y))]
    num = 0
    for i in range(len(tmp)):
        # print i+1, y[i], tmp[i], tmp[i]/y[i]
        if tmp[i]/y[i] >= 0.5:
            num += 1
    print("rate of the difference between y and y_pred more than 0.5: %f" % (float(num)/len(y)))
    return sum(tmp)/sum(y)

def rmse(y, y_pred):
    tmp = [(y[i] - y_pred[i]) * (y[i] - y_pred[i]) for i in range(len(y))]
    import math
    return math.pow(sum(tmp)/len(tmp), 0.5)

def wmae(preds, dtrain):
    y = dtrain.get_label()
    # tmp = [(y[i] - preds[i]) * (y[i] - preds[i]) for i in range(len(y))]
    tmp = [abs(y[i] - preds[i]) for i in range(len(y))]
    return 'wmae', sum(tmp)/sum(y) 

def compare(file1, file2):
    data1 = pd.read_csv(file1).values
    result1 = data1[:, -1]
    data2 = pd.read_csv(file2).values
    result2 = data2[:, -1]
    del data1, data2
    print("the WMAE between %s and %s is %f" % (file1, file2, get_wmae(result2, result1)))

def pre():
    t_order = pd.read_csv('./data/t_order.csv')
    t_sales_sum = pd.read_csv('./data/t_sales_sum.csv')
    t_ads = pd.read_csv('./data/t_ads.csv')

    shop_id = list(set(t_order['shop_id']))
    sales = []
    data = np.zeros((len(shop_id), 5))

    for i, id in enumerate(shop_id):
        order = t_order[t_order['shop_id'].isin([id])].values
        for j in range(len(order)):
            if order[j][0] >= '2017-01-01' and order[j][0] <= '2017-03-31':
                data[i][0] += order[j][1] # sale_amt
                data[i][1] += order[j][2] # offer_amt
                data[i][2] += order[j][6] # rtn_amt
        ads = t_ads[t_ads['shop_id'].isin([id])].values
        for j in range(len(ads)):
            if ads[j][0] >= '2017-01-01' and ads[j][0] <= '2017-03-31':
                data[i][3] += ads[j][1] # charge
                data[i][4] += ads[j][2] # consume

        tmp = t_sales_sum[t_sales_sum['shop_id'].isin([i+1]) & t_sales_sum['dt'].isin(['2016-12-31'])].values
        sales.append(tmp[0][2])
        print i+1, sales[i], data[i][0], data[i][1], data[i][2], data[i][3], data[i][4], (data[i][0]-data[i][1]-data[i][2]-data[i][4])

    
def get_feature(num):
    times = [['2016-08-31', '2016-08-01', '2016-08-31', 1],
             ['2016-09-30', '2016-09-01', '2016-09-30', 1],
             ['2016-10-31', '2016-10-01', '2016-10-31', 1],
             ['2016-11-30', '2016-11-01', '2016-11-30', 1],
             ['2016-12-31', '2016-12-01', '2016-12-31', 1],
             ['2017-01-31', '2017-01-01', '2017-01-31', 1],
             ['2017-01-31', '2016-12-01', '2017-01-31', 1],
             ['2016-12-31', '2016-11-01', '2017-12-31', 1],
             ['2016-11-30', '2016-10-01', '2016-11-30', 1],
             ['2016-10-31', '2016-09-01', '2016-10-31', 1],
             ['2016-09-30', '2016-08-01', '2016-09-30', 1],
             ['2016-10-31', '2016-08-01', '2016-10-31', 1],
             ['2016-11-30', '2016-09-01', '2016-11-30', 1],
             ['2016-12-31', '2016-10-01', '2016-12-31', 1],]
             # ['2017-01-31', '2016-11-01', '2016-01-31', 1]]

    data = []
    t_order = pd.read_csv('./data/t_order.csv')
    t_sales_sum = pd.read_csv('./data/t_sales_sum.csv')
    t_ads = pd.read_csv('./data/t_ads.csv')
    t_comment = pd.read_csv('./data/t_comment.csv')
    t_product = pd.read_csv('./data/t_product.csv')
    data_product = t_product.values

    dic_pid = {}
    dic_brand = {}
    dic_cate = {}
    for i in range(len(data_product)):
        if data_product[i, 2] not in dic_brand:
            dic_brand[data_product[i, 2]] = 0
        dic_brand[data_product[i, 2]] += 1
        if data_product[i, 3] not in dic_cate:
            dic_cate[data_product[i, 3]] = 0
        dic_cate[data_product[i, 3]] += 1
        dic_pid[data_product[i, 5]] =[data_product[i, 2], data_product[i, 3]]


    # shop_id = range(1, 3)
    for id in tqdm(range(1, 3001)):
        for t in times:
            order = t_order[t_order['shop_id'].isin([id])].values
            sale_amt = offer_amt = offer_cnt = rtn_cnt = rtn_amt = ord_cnt = user_cnt = 0.0
            pid = []
            for j in range(len(order)):
                if order[j][0] >= t[1] and order[j][0] <= t[2]:
                    sale_amt += order[j][1] # sale_amt
                    offer_amt += order[j][2] # offer_amt
                    offer_cnt += order[j][3] # offer_cnt
                    rtn_cnt += order[j][5] # rtn_cnt
                    rtn_amt += order[j][6] # rtn_amt
                    ord_cnt += order[j][7] # ord_cnt
                    user_cnt += order[j][8] # user_cnt
                    pid.append(order[j][8]) # pid
            pid = list(set(pid))
            # brand = [dic_pid[p][0] for p in pid]
            brand = []
            cate = []
            for p in pid:
                if p in dic_pid:
                    brand.append(dic_pid[p][0])
                    cate.append(dic_pid[p][1])
            brand = len(list(set(brand)))
            # cate = [dic_pid[p][1] for p in pid]
            cate = len(list(set(cate)))
            pid = len(pid)
            ads = t_ads[t_ads['shop_id'].isin([id])].values
            ads_charge = ads_consume = 0.0
            for j in range(len(ads)):
                ads_charge += ads[j][1] # charge
                ads_consume += ads[j][2] # consume

            comment = t_comment[t_comment['shop_id'].isin([id])].values
            bad_num = mid_num = good_num = dis_num = cmmt_num = 0
            for j in range(len(comment)):
                if comment[j][0] >= t[1] and comment[j][0] <= t[2]:
                    bad_num += comment[j][1]
                    mid_num += comment[j][5]
                    good_num += comment[j][4]
                    dis_num += comment[j][3]
                    cmmt_num += comment[j][2]
            # print bad_num, mid_num, good_num, cmmt_num
            if cmmt_num == 0:
                bad_rate = mid_rate = good_rate = 0.0
            else:
                bad_rate = float(bad_num) / cmmt_num 
                mid_rate = float(mid_num) / cmmt_num 
                good_rate = float(good_num) / cmmt_num 
            sales_sum = t_sales_sum[t_sales_sum['shop_id'].isin([id])].values
            sale_sum = 0.
            for j in range(len(sales_sum)):
                if sales_sum[j][0] == t[0]:
                    sale_sum = sales_sum[j][2]
                    break
            data.append([pid, sale_amt, offer_amt, offer_cnt, rtn_cnt, rtn_amt, ord_cnt, user_cnt, ads_charge, ads_consume, bad_rate, mid_rate, good_rate, dis_num, cmmt_num, brand, cate, sale_sum])
    print np.shape(data)
    columns = ['pid', 'sale_amt', 'offer_amt', 'offer_cnt', 'rtn_cnt', 'rtn_amt', 'ord_cnt', 'user_cnt', 'ads_charge', 'ads_consume', 'bad_rate',
               'mid_rate', 'good_rate', 'dis_num', 'cmmt_num', 'brand', 'cate', 'sale_sum']
    df = pd.DataFrame(data, columns=columns)
    df.to_csv('./1125/pre_%d_17.csv' % num, index=False)

def train(filename):
    data = pd.read_csv(filename).values
    np.random.shuffle(data)
    split = len(data) * 5 / 6
    X_train = data[0:split, 0:-1]
    y_train = data[0:split, -1]
    X_test = data[split:, 0:-1]
    y_test = data[split:, -1]
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    xgb_train = xgb.DMatrix(X_train, label=y_train)
    xgb_val = xgb.DMatrix(X_val, label=y_val)
    params = {'booster': 'gbtree',
              'objective': 'reg:linear',
              'gamma': 0.1,
              'max_depth': 10, 'lambda': 2, 'subsample': 0.9,
              'colsample_bytree': 0.8, 'min_child_weights': 3, # 'eval_metric': 'rmse',
              'silent': 1, 'eta': 0.05, 'seed': 50, 'nthread': 7}
    # params = {'objective': "reg:linear", 
    #           'eta': 0.15, 
    #           'max_depth': 10, 
    #           'subsample': 0.7, 
    #           'colsample_bytree': 0.7, 
    #           'silent': 1}
    plst = list(params.items())
    num_round = 1000
    watchlist = [(xgb_train, 'train'), (xgb_val, 'val')]

    # train
    model = xgb.train(plst, xgb_train, num_round, watchlist, feval=wmae, early_stopping_rounds=20)
    # model.save_model('xgb.model')
    print("best ntree limit {}".format(model.best_ntree_limit))
    test = xgb.DMatrix(X_test, label=y_test)
    y_pred = model.predict(test)
    # print y_pred
    print np.shape(y_pred)
    print "The wmae is: %f" % get_wmae(y_test, y_pred)
    print "The %s is: %f" % wmae(y_pred, test)
    print "The rmse is %f" % rmse(y_test, y_pred)
    return model

def test_feature():
    times = [['2017-04-30', '2017-04-01', '2017-04-30', 1],]
             # ['2017-04-30', '2017-03-01', '2017-04-30', 2],]
             # ['2017-04-30', '2017-02-01', '2017-04-30', 3]]
    data = []
    t_order = pd.read_csv('./data/t_order.csv')
    t_sales_sum = pd.read_csv('./data/t_sales_sum.csv')
    t_ads = pd.read_csv('./data/t_ads.csv')
    t_comment = pd.read_csv('./data/t_comment.csv')
    t_product = pd.read_csv('./data/t_product.csv')
    data_product = t_product.values

    dic_pid = {}
    dic_brand = {}
    dic_cate = {}
    for i in range(len(data_product)):
        if data_product[i, 2] not in dic_brand:
            dic_brand[data_product[i, 2]] = 0
        dic_brand[data_product[i, 2]] += 1
        if data_product[i, 3] not in dic_cate:
            dic_cate[data_product[i, 3]] = 0
        dic_cate[data_product[i, 3]] += 1
        dic_pid[data_product[i, 5]] = [data_product[i, 2], data_product[i, 3]]

    for id in tqdm(range(1, 3001)):
        for t in times:
            order = t_order[t_order['shop_id'].isin([id])].values
            sale_amt = offer_amt = offer_cnt = rtn_cnt = rtn_amt = ord_cnt = user_cnt = 0.0
            pid = []
            for j in range(len(order)):
                if order[j][0] >= t[1] and order[j][0] <= t[2]:
                    sale_amt += order[j][1] # sale_amt
                    offer_amt += order[j][2] # offer_amt
                    offer_cnt += order[j][3] # offer_cnt
                    rtn_cnt += order[j][5] # rtn_cnt
                    rtn_amt += order[j][6] # rtn_amt
                    ord_cnt += order[j][7] # ord_cnt
                    user_cnt += order[j][8] # user_cnt
                    pid.append(order[j][8]) # pid
            pid = list(set(pid))
            brand = []
            cate = []
            for p in pid:
                if p in dic_pid:
                    brand.append(dic_pid[p][0])
                    cate.append(dic_pid[p][1])
            brand = len(list(set(brand)))
            # cate = [dic_pid[p][1] for p in pid]
            cate = len(list(set(cate)))
            pid = len(pid)
            ads = t_ads[t_ads['shop_id'].isin([id])].values
            ads_charge = ads_consume = 0.0
            for j in range(len(ads)):
                ads_charge += ads[j][1]  # charge
                ads_consume += ads[j][2]  # consume

            comment = t_comment[t_comment['shop_id'].isin([id])].values
            bad_num = mid_num = good_num = dis_num = cmmt_num = 0
            for j in range(len(comment)):
                if comment[j][0] >= t[1] and comment[j][0] <= t[2]:
                    bad_num += comment[j][1]
                    mid_num += comment[j][5]
                    good_num += comment[j][4]
                    dis_num += comment[j][3]
                    cmmt_num += comment[j][2]
            
            # print bad_num, mid_num, good_num, cmmt_num
            if cmmt_num == 0:
                bad_rate = mid_rate = good_rate = 0.0
            else:
                bad_rate = float(bad_num) / cmmt_num 
                mid_rate = float(mid_num) / cmmt_num 
                good_rate = float(good_num) / cmmt_num

	    data.append([pid, sale_amt, offer_amt, offer_cnt, rtn_cnt, rtn_amt, ord_cnt, user_cnt, ads_charge, ads_consume, bad_rate, mid_rate, good_rate, dis_num, cmmt_num, brand, cate])
    print np.shape(data)
    columns = ['pid', 'sale_amt', 'offer_amt', 'offer_cnt', 'rtn_cnt', 'rtn_amt', 'ord_cnt', 'user_cnt', 'ads_charge', 'ads_consume', 'bad_rate',
               'mid_rate', 'good_rate', 'dis_num', 'cmmt_num', 'brand', 'cate']
    df = pd.DataFrame(data, columns=columns)
    df.to_csv('./1125/pre_test_1_17.csv', index=False)

def predict(filenames):
    model = train(filename=filenames[0])

    X_test1 = pd.read_csv(filenames[1]).values
    test1 = xgb.DMatrix(X_test1)
    y_pred1 = model.predict(test1)
    print('y_pred1', np.shape(y_pred1))

    X_test2 = pd.read_csv(filenames[2]).values
    test2 = xgb.DMatrix(X_test2)
    y_pred2 = model.predict(test2)
    print('y_pred2', np.shape(y_pred2))

    X_test3 = pd.read_csv(filenames[3]).values
    test3 = xgb.DMatrix(X_test3)
    y_pred3 = model.predict(test3)
    print('y_pred3', np.shape(y_pred3))

    y_pred = []
    for i in range(3000):
        y_pred.append([i+1, sum([y_pred1[i], y_pred2[i], y_pred3[i]])/3])
    df = pd.DataFrame(y_pred)
    df.to_csv('report.csv', index=False, header=False)
    # print "The wmae is: %f" % wmae(y_test, y_pred)


if __name__ == '__main__':
    start = time.time()
    # get_feature(num=42000)
    # test_feature()
    # train(filename='./pre/42000_15.csv')
    # filenames = ['./1125/pre_42000_17.csv', './1125/pre_test_1_17.csv', './1125/pre_test_2_17.csv', './1125/pre_test_3_17.csv']
    # predict(filenames)
    compare('./report/report_474608.csv', './report/report_449257.csv')
    compare('./report/report_476405.csv', './report/report_474608.csv')
    compare('./report/report_478967.csv', './report/report_474608.csv')
    compare('./report/report_52171.csv', './report/report_474608.csv')
    end = time.time()
    print 'Time consumption: %.2f sec...' % (end - start)
