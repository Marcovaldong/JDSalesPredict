import time
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.utils import shuffle
import xgboost as xgb
from sklearn.model_selection import train_test_split
# import threading
# from keras.models import Input, Model
# from keras.layers import Dense
# from keras import optimizers
def days(start_dt, end_dt):
    dt1 = [int(i) for i in start_dt.split('-')]
    dt2 = [int(i) for i in end_dt.split('-')]
    dt1 = datetime.date(dt1[0], dt1[1], dt1[2])
    dt2 = datetime.date(dt2[0], dt2[1], dt2[2])
    ret = dt2 - dt1
    return ret.days + 1

def get_days(start_dt, end_dt, on_dt, off_dt):
    if start_dt >= on_dt and end_dt <= off_dt:
        dt1 = [int(i) for i in start_dt.split('-')]
        dt2 = [int(i) for i in end_dt.split('-')]
    elif start_dt < on_dt and end_dt <= off_dt:
        dt1 = [int(i) for i in on_dt.split('-')]
        dt2 = [int(i) for i in end_dt.split('-')]
    elif start_dt >= on_dt and end_dt > off_dt:
        dt1 = [int(i) for i in start_dt.split('-')]
        dt2 = [int(i) for i in off_dt.split('-')]
    elif start_dt < on_dt and end_dt > off_dt:
        dt1 = [int(i) for i in on_dt.split('-')]
        dt2 = [int(i) for i in off_dt.split('-')]
    else:
        return 0
    dt1 = datetime.date(dt1[0], dt1[1], dt1[2])
    dt2 = datetime.date(dt2[0], dt2[1], dt2[2])
    ret = dt2 - dt1
    return ret.days + 1


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

def pre_train(num1, num2):
    # times = [['2016-08-31', '2016-08-01', '2016-08-31', '2016-09-01', '2016-11-29'],
    #          ['2016-09-30', '2016-09-01', '2016-09-30', '2016-10-01', '2016-12-29'],
    #          ['2016-10-31', '2016-10-01', '2016-10-31', '2016-11-01', '2017-01-29'],
    #          ['2016-11-30', '2016-11-01', '2016-11-30', '2016-12-01', '2017-02-28'],
    #          ['2016-12-31', '2016-12-01', '2016-12-31', '2017-01-01', '2017-03-31'],
    #          ['2017-01-31', '2017-01-01', '2017-01-31', '2017-02-01', '2017-05-01'],
    #          ['2017-01-31', '2016-12-01', '2017-01-31', '2017-02-01', '2017-05-01'],
    #          ['2016-12-31', '2016-11-01', '2017-12-31', '2017-01-01', '2017-03-31'],
    #          ['2016-11-30', '2016-10-01', '2016-11-30', '2016-12-01', '2017-02-28'],
    #          ['2016-10-31', '2016-09-01', '2016-10-31', '2016-11-01', '2017-01-29'],
    #          ['2016-09-30', '2016-08-01', '2016-09-30', '2016-10-01', '2016-12-29'],
    #          ['2016-10-31', '2016-08-01', '2016-10-31', '2016-11-01', '2017-01-29'],
    #          ['2016-11-30', '2016-09-01', '2016-11-30', '2016-12-01', '2017-02-28'],
    #          ['2016-12-31', '2016-10-01', '2016-12-31', '2017-01-01', '2017-03-31'],
    #          ['2017-01-31', '2016-11-01', '2017-01-31', '2017-02-01', '2017-05-01'],
    times = [['2017-01-31', '2016-10-01', '2017-01-31', '2017-02-01', '2017-05-01'],
             ['2016-12-31', '2016-09-01', '2016-12-31', '2017-01-01', '2017-03-31'],
             ['2016-11-30', '2016-08-03', '2016-11-30', '2016-12-01', '2017-02-28'],
             ['2017-01-31', '2016-09-01', '2017-01-31', '2017-02-01', '2017-05-01'],
             ['2016-12-31', '2016-08-03', '2016-12-31', '2017-01-01', '2017-03-31'],
             ['2017-01-31', '2016-08-03', '2017-01-31', '2017-02-01', '2017-05-01'],]

    t_order = pd.read_csv('./data/t_order.csv')
    t_sales_sum = pd.read_csv('./data/t_sales_sum.csv')
    t_ads = pd.read_csv('./data/t_ads.csv')
    t_comment = pd.read_csv('./data/t_comment.csv')
    t_product = pd.read_csv('./data/t_product.csv')

    # data_product = t_product.values
    # pid_brabd_cate = {}
    # dic_brand = {}
    # dic_cate = {}
    # for i in range(len(data_product)):
    #     if data_product[i, 2] not in dic_brand:
    #         dic_brand[data_product[i, 2]] = 0
    #     dic_brand[data_product[i, 2]] += 1
    #     if data_product[i, 3] not in dic_cate:
    #         dic_cate[data_product[i, 3]] = 0
    #     dic_cate[data_product[i, 3]] += 1
    #     pid_brabd_cate[data_product[i, 5]] = [data_product[i, 2], data_product[i, 3]]
    data = []
    for shop_id in tqdm(range(num1, num2)):
        order_shop = t_order[t_order['shop_id'].isin([shop_id])]
        comment_shop = t_comment[t_comment['shop_id'].isin([shop_id])]
        product_shop = t_product[t_product['shop_id'].isin([shop_id])]
        for n, t in enumerate(times):
            # print shop_id, n
            order = order_shop.loc[(order_shop['ord_dt'] >= t[1]) & (order_shop['ord_dt'] <= t[2])].values
            dic_pid = {}
            for j in range(len(order)):
                if order[j][8] not in dic_pid:
                    dic_pid[order[j][8]] = [0, 0, 0, 0, 0, 0, 0]
                tmp = [order[j][1], order[j][2], order[j][3], order[j][5], order[j][6], order[j][7], order[j][9]]
                dic_pid[order[j][8]] = [dic_pid[order[j][8]][k] + tmp[k] for k in range(7)]
            sale_amt = offer_amt = offer_cnt = rtn_cnt = rtn_amt = ord_cnt = user_cnt = 0.0
            for pid in dic_pid:
                tmp = product_shop[product_shop['pid'].isin([pid])].values
                # print tmp
                if len(tmp)>0:
                    on_dt, off_dt = tmp[0][0], tmp[0][1]
                    if type(off_dt) == float:
                        off_dt = '2017-08-31'
                    days1 = get_days(t[1], t[2], on_dt, off_dt)
                    days2 = get_days(t[1], t[2], on_dt, off_dt)
                    tmp = days2 / float(days1)
                else:
                    days1 = days(t[1], t[2])
                    days2 = days(t[3], t[4])
                    tmp = days2 / float(days1)
                dic_pid[pid] = [k * tmp for k in dic_pid[pid]]
                sale_amt += dic_pid[pid][0]
                offer_amt += dic_pid[pid][1]
                offer_cnt += dic_pid[pid][2]
                rtn_cnt += dic_pid[pid][3]
                rtn_amt += dic_pid[pid][4]
                ord_cnt += dic_pid[pid][5]
                user_cnt += dic_pid[pid][6]
            pid_num = len(dic_pid)
            # comment = comment_shop.loc[comment_shop['create_dt'] >= t[1]]
            # comment = comment.loc[comment['create_dt'] <= t[2]].values
            comment = comment_shop.loc[(comment_shop['create_dt'] >= t[1]) & (comment_shop['create_dt'] <= t[2])].values
            bad_num = mid_num = good_num = dis_num = cmmt_num = 0
            for j in range(len(comment)):
                bad_num += comment[j][1]
                mid_num += comment[j][5]
                good_num += comment[j][4]
                dis_num += comment[j][3]
                cmmt_num += comment[j][2]

            if cmmt_num == 0:
                bad_rate = mid_rate = good_rate = 0.0
            else:
                bad_rate = float(bad_num) / cmmt_num
                mid_rate = float(mid_num) / cmmt_num
                good_rate = float(good_num) / cmmt_num
            cmmt_num = cmmt_num * 90.0/ days(t[1], t[2])

            ads = t_ads[t_ads['shop_id'].isin([shop_id])].values
            ads_charge = ads_consume = 0.0
            for j in range(len(ads)):
                ads_charge += ads[j][1]  # charge
                ads_consume += ads[j][2]  # consume

            sales_sum = t_sales_sum[t_sales_sum['shop_id'].isin([shop_id])].values
            sale_sum = 0.
            for j in range(len(sales_sum)):
                if sales_sum[j][0] == t[0]:
                    sale_sum = sales_sum[j][2]
                    break
            data.append([shop_id, t[0], t[1], t[2], t[3], t[4], pid_num, sale_amt, offer_amt, offer_cnt, rtn_cnt, rtn_amt, ord_cnt, user_cnt, ads_charge, ads_consume, bad_rate, mid_rate, good_rate, dis_num, cmmt_num, sale_sum])
    print np.shape(data)
    columns = ['shop_id', 'dt0', 'dt1', 'dt2', 'dt3', 'dt4', 'pid_num', 'sale_amt', 'offer_amt', 'offer_cnt', 'rtn_cnt', 'rtn_amt', 'ord_cnt', 'user_cnt', 'ads_charge', 'ads_consume', 'bad_rate',
               'mid_rate', 'good_rate', 'dis_num', 'cmmt_num', 'sale_sum']
    df = pd.DataFrame(data, columns=columns)
    df.to_csv('./1128/pre_%d_%d.csv' % (num1, num2-1), index=False)

def pre_test(times, filename):
    # times = [  # ['2017-04-30', '2017-04-01', '2017-04-30', '2017-05-01', '2017-07-29'],]
               # ['2017-04-30', '2017-03-01', '2017-04-30', '2017-05-01', '2017-07-29'],]
               # ['2017-04-30', '2017-02-01', '2017-04-30', '2017-05-01', '2017-07-29']]
    data = []
    t_order = pd.read_csv('./data/t_order.csv')
    # t_sales_sum = pd.read_csv('./data/t_sales_sum.csv')
    t_ads = pd.read_csv('./data/t_ads.csv')
    t_comment = pd.read_csv('./data/t_comment.csv')
    t_product = pd.read_csv('./data/t_product.csv')

    for shop_id in tqdm(range(1, 3001)):
        order_shop = t_order[t_order['shop_id'].isin([shop_id])]
        comment_shop = t_comment[t_comment['shop_id'].isin([shop_id])]
        product_shop = t_product[t_product['shop_id'].isin([shop_id])]
        for n, t in enumerate(times):
            # print shop_id, n
            order = order_shop.loc[(order_shop['ord_dt'] >= t[1]) & (order_shop['ord_dt'] <= t[2])].values
            dic_pid = {}
            for j in range(len(order)):
                if order[j][8] not in dic_pid:
                    dic_pid[order[j][8]] = [0, 0, 0, 0, 0, 0, 0]
                tmp = [order[j][1], order[j][2], order[j][3], order[j][5], order[j][6], order[j][7], order[j][9]]
                dic_pid[order[j][8]] = [dic_pid[order[j][8]][k] + tmp[k] for k in range(7)]
            sale_amt = offer_amt = offer_cnt = rtn_cnt = rtn_amt = ord_cnt = user_cnt = 0.0
            for pid in dic_pid:
                tmp = product_shop[product_shop['pid'].isin([pid])].values
                # print tmp
                if len(tmp)>0:
                    on_dt, off_dt = tmp[0][0], tmp[0][1]
                    if type(off_dt) == float:
                        off_dt = '2017-08-31'
                    days1 = get_days(t[1], t[2], on_dt, off_dt)
                    days2 = get_days(t[1], t[2], on_dt, off_dt)
                    tmp = days2 / float(days1)
                else:
                    days1 = days(t[1], t[2])
                    days2 = days(t[3], t[4])
                    tmp = days2 / float(days1)
                dic_pid[pid] = [k * tmp for k in dic_pid[pid]]
                sale_amt += dic_pid[pid][0]
                offer_amt += dic_pid[pid][1]
                offer_cnt += dic_pid[pid][2]
                rtn_cnt += dic_pid[pid][3]
                rtn_amt += dic_pid[pid][4]
                ord_cnt += dic_pid[pid][5]
                user_cnt += dic_pid[pid][6]
            pid_num = len(dic_pid)

            comment = comment_shop.loc[(comment_shop['create_dt'] >= t[1]) & (comment_shop['create_dt'] <= t[2])].values
            bad_num = mid_num = good_num = dis_num = cmmt_num = 0
            for j in range(len(comment)):
                bad_num += comment[j][1]
                mid_num += comment[j][5]
                good_num += comment[j][4]
                dis_num += comment[j][3]
                cmmt_num += comment[j][2]

            if cmmt_num == 0:
                bad_rate = mid_rate = good_rate = 0.0
            else:
                bad_rate = float(bad_num) / cmmt_num
                mid_rate = float(mid_num) / cmmt_num
                good_rate = float(good_num) / cmmt_num
            cmmt_num = cmmt_num * 90.0 / days(t[1], t[2])
            ads = t_ads[t_ads['shop_id'].isin([shop_id])].values
            ads_charge = ads_consume = 0.0
            for j in range(len(ads)):
                ads_charge += ads[j][1]  # charge
                ads_consume += ads[j][2]  # consume

            data.append([t[0], t[1], t[2], t[3], t[4], pid_num, sale_amt, offer_amt, offer_cnt, rtn_cnt, rtn_amt, ord_cnt, user_cnt, ads_charge, ads_consume, bad_rate, mid_rate, good_rate, dis_num, cmmt_num])
    print np.shape(data)
    columns = ['dt0', 'dt1', 'dt2', 'dt3', 'dt4', 'pid_num', 'sale_amt', 'offer_amt', 'offer_cnt', 'rtn_cnt', 'rtn_amt', 'ord_cnt', 'user_cnt', 'ads_charge', 'ads_consume', 'bad_rate',
               'mid_rate', 'good_rate', 'dis_num', 'cmmt_num']
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(filename, index=False)

def about_holiday():
    '''12.01'''
    holi = pd.read_csv('./data/holiday.csv').values
    isholiday = {}
    for i in range(len(holi)):
        isholiday[holi[i][0]] = holi[i][1]
    workdays = 177; weekend = 68; holidays = 26
    t_order = pd.read_csv('./data/t_order.csv')
    ord_dt = list(t_order['ord_dt'])
    attribute = []
    for i in range(len(ord_dt)):
        attribute.append(isholiday[ord_dt[i]])
    attribute = np.array(attribute)
    attribute = attribute.reshape((len(attribute), 1))
    order = t_order.values
    columns = list(t_order.columns)
    columns.append('isholi')
    order = np.concatenate([order, attribute], axis=1)
    t_order = pd.DataFrame(order, columns=columns)
    # print t_order.head()

    data = []
    for shop_id in tqdm(range(1, 3001)):
        order_shop = t_order[t_order['shop_id'] == shop_id]
        statics = order_shop.groupby('isholi', as_index=False).agg({'ord_cnt': 'sum', 'sale_amt': 'sum', 'user_cnt': 'sum',
                                                                   'rtn_amt': 'sum', 'rtn_cnt': 'sum', 'offer_amt': 'sum', 'offer_cnt': 'sum'})
        # print 'shop_id', shop_id
        # print statics.head()
        statics = statics.values
        items = [np.array([0, 0, 0, 0, 0, 0, 0]), np.array([0, 0, 0, 0, 0, 0, 0]), np.array([0, 0, 0, 0, 0, 0, 0]), np.array([0, 0, 0, 0, 0, 0, 0])]
        items[0] = statics[0][1:] / workdays
        items[1] = statics[1][1:] / weekend
        items[2] = statics[2][1:] / holidays
        try:
            items[3] = statics[3][1:]
        except:
            continue

        # print items[0]
        # print items[1]
        # print items[2]
        data.append([shop_id, 0] + list(items[0]))
        data.append([shop_id, 1] + list(items[1]))
        data.append([shop_id, 2] + list(items[2]))
        data.append([shop_id, 3] + list(items[3]))
    columns = ['shop_id', 'holi', 'ord_cnt', 'sale_amt', 'user_cnt', 'rtn_amt', 'rtn_cnt', 'offer_amt', 'offer_cnt']
    df = pd.DataFrame(data, columns=columns)
    df.to_csv('./data/static_holiday_1111.csv', index=False)

def add_3m_sales(filename1='./1201/63000_17_reg.csv', filename2='./1202/63000_18_reg.csv'):
    '''12.02'''
    feats = pd.read_csv(filename1)
    shop_ids = list(feats['shop_id'])
    dt0 = list(feats['dt0'])
    columns = list(feats.columns) + ['first3m_sales']
    feats = feats.values
    t_sales_sum = pd.read_csv('./data/t_sales_sum.csv')
    dic = {'2016-08-31': '2016-06-30', '2016-09-30': '2016-06-30', '2016-10-31': '2016-07-31', '2016-11-30': '2016-08-31',
           '2016-12-31': '2016-09-30', '2017-01-31': '2016-10-31', '2017-04-30': '2017-01-31'}
    first3m_sales = []
    for i in tqdm(range(len(shop_ids))):
        tmp = t_sales_sum[(t_sales_sum['shop_id'] == shop_ids[i]) & (t_sales_sum['dt'] == dic[dt0[i]])].values
        first3m_sales.append(tmp[0][2])
    first3m_sales = np.array(first3m_sales).reshape((len(shop_ids), 1))
    feats = np.concatenate([feats, first3m_sales], axis=1)
    df = pd.DataFrame(feats, columns=columns)
    df.to_csv(filename2, index=False)





def process():
    # data = pd.read_csv('./1127/id_dt_45000_15.csv')
    # columns = list(data.columns)
    # start_dt = list(data['dt1'])
    # end_dt = list(data['dt2'])
    # data = data.values
    # for i in range(len(start_dt)):
    #     data[i][19] = float(data[i][19]) * 90 / days(start_dt[i], end_dt[i])
    # df = pd.DataFrame(data, columns=columns)
    # df.to_csv('./1127/45000_15.csv', index=False)

    for k in range(1, 4):
        data = pd.read_csv('./1127/pre_test_%d.csv' % k)
        columns = ['shop_id'] + list(data.columns)
        start_dt = list(data['dt1'])
        end_dt = list(data['dt2'])
        data = data.values
        shop_id = []
        for i in range(len(start_dt)):
            data[i][18] = float(data[i][18]) * 90 / days(start_dt[i], end_dt[i])
            shop_id.append([[i+1]])
        shop_id = np.concatenate(shop_id, axis=0)
        data = np.concatenate([shop_id, data], axis=1)
        df = pd.DataFrame(data, columns=columns)
        df.to_csv('./1127/test_%d.csv' % k, index=False)

def get_feature():
    pre_train(1, 3001)
    pre_test()

def train(filename, num_round, early_stopping_rounds):
    data = pd.read_csv(filename).values
    np.random.shuffle(data)
    split = len(data) * 5 / 6
    X_train = data[0:split, 7:-1]
    y_train = data[0:split, -1]
    X_test = data[split:, 7:-1]
    y_test = data[split:, -1]
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.30, random_state=42)
    xgb_train = xgb.DMatrix(X_train, label=y_train)
    xgb_val = xgb.DMatrix(X_val, label=y_val)
    params = {'booster': 'gbtree',
              'objective': 'reg:linear',
              'eval_metric': 'rmse',
              'gamma': 0.2,
              'max_depth': 10, 'lambda': 2, 'subsample': 0.7, 'alpha': 1,
              # 'colsample_bytree': 0.7, 'colsample_bylevel': 0.7, 'min_child_weights': 3, # 'eval_metric': 'rmse',
              'silent': 1, 'eta': 0.03, 'seed': 50, }
    plst = list(params.items())
    num_round = num_round
    watchlist = [(xgb_train, 'train'), (xgb_val, 'val')]

    # train
    model = xgb.train(plst, xgb_train, num_round, watchlist, feval=wmae, early_stopping_rounds=early_stopping_rounds)
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

def train_Kfold(filename, num_round, early_stopping_rounds):
    '''
    12.01
    '''
    params = {'booster': 'gbtree',
              'objective': 'reg:linear',
              'eval_metric': 'rmse',
              'gamma': 0.2,
              'max_depth': 10, 'lambda': 2, 'subsample': 0.7, 'alpha': 1,
              # 'colsample_bytree': 0.7, 'colsample_bylevel': 0.7, 'min_child_weights': 3, # 'eval_metric': 'rmse',
              'silent': 1, 'eta': 0.03, 'seed': 50, }
    plst = list(params.items())
    num_round = num_round
    precision = []
    data = pd.read_csv(filename).values
    np.random.shuffle(data)
    num = len(data)
    fold = num / 10
    for i in range(1, 11):
        test = data[(i-1)*fold:i*fold, :]
        X_test = test[:, 7:-1]
        y_test = test[:, -1]
        train = np.concatenate([data[:(i-1)*fold, :], data[i*fold:, :]], axis=0)
        np.random.shuffle(train)
        X_train = train[:, 7:-1]
        y_train = train[:, -1]
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
        xgb_train = xgb.DMatrix(X_train, label=y_train)
        xgb_val = xgb.DMatrix(X_val, label=y_val)
        watchlist = [(xgb_train, 'train'), (xgb_val, 'val')]
        model = xgb.train(plst, xgb_train, num_round, watchlist, feval=wmae,
                          early_stopping_rounds=early_stopping_rounds)
        print("best ntree limit {}".format(model.best_ntree_limit))
        test = xgb.DMatrix(X_test, label=y_test)
        y_pred = model.predict(test)
        preci = get_wmae(y_test, y_pred)
        precision.append(preci)
        print "The wmae is: %f" % preci
        print "The %s is: %f" % wmae(y_pred, test)
        print "The rmse is %f" % rmse(y_test, y_pred)
        model.save_model('./model/xgb_%s.model' % str(preci))
        print "Finished training the %dth model..." % i
    print precision
    best = min(precision)
    bst = xgb.Booster({'nthread': 4})
    bst.load_model('./model/xgb_%s.model' % str(best))
    return bst


def predict(model, filenames):
    # model = train(filename=filenames[0], num_round=1000, early_stopping_rounds=20)

    X_test1 = pd.read_csv(filenames[1]).values
    X_test1 = X_test1[:, 7:]
    test1 = xgb.DMatrix(X_test1)
    y_pred1 = model.predict(test1)
    print('y_pred1', np.shape(y_pred1))

    X_test2 = pd.read_csv(filenames[2]).values
    X_test2 = X_test2[:, 7:]
    test2 = xgb.DMatrix(X_test2)
    y_pred2 = model.predict(test2)
    print('y_pred2', np.shape(y_pred2))

    X_test3 = pd.read_csv(filenames[3]).values
    X_test3 = X_test3[:, 7:]
    test3 = xgb.DMatrix(X_test3)
    y_pred3 = model.predict(test3)
    print('y_pred3', np.shape(y_pred3))

    y_pred = []
    for i in range(3000):
        y_pred.append([i+1, sum([y_pred2[i], y_pred3[i]])/2]) # y_pred1[i],
    df = pd.DataFrame(y_pred)
    df.to_csv('report_1202.csv', index=False, header=False)
    # print "The wmae is: %f" % wmae(y_test, y_pred)


if __name__ == '__main__':
    start = time.time()
    # pre_train(1801, 1901)
    # pre_train(1901, 2001)
    # pre_test(times=[['2017-04-30', '2017-04-01', '2017-04-30', '2017-05-01', '2017-07-29']], filename='./1127/pre_test_1.csv')
    # pre_test(times=[['2017-04-30', '2017-03-01', '2017-04-30', '2017-05-01', '2017-07-29']], filename='./1127/pre_test_2.csv')
    # pre_test(times=[['2017-04-30', '2017-02-01', '2017-04-30', '2017-05-01', '2017-07-29']], filename='./1127/pre_test_3.csv')

    # about_holiday()
    # add_3m_sales(filename1='./1201/63000_17_reg.csv', filename2='./1202/63000_18_reg.csv')
    # add_3m_sales(filename1='./1201/test_1_17_reg.csv', filename2='./1202/test_1_18_reg.csv')
    # add_3m_sales(filename1='./1201/test_2_17_reg.csv', filename2='./1202/test_2_18_reg.csv')
    # add_3m_sales(filename1='./1201/test_3_17_reg.csv', filename2='./1202/test_3_18_reg.csv')

    # model = train(filename='./1201/63000_17_reg.csv', num_round=1000, early_stopping_rounds=20)
    model = train_Kfold(filename='./1202/60000_18_reg.csv', num_round=1000, early_stopping_rounds=10)
    filenames = ['./1202/60000_18_reg.csv', './1202/test_1_18_reg.csv', './1202/test_2_18_reg.csv', './1202/test_3_18_reg.csv']
    predict(model, filenames)


    end = time.time()
    print 'Time consumption: %.2f sec...' % (end - start)
