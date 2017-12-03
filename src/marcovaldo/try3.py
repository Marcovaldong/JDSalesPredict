import time
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.utils import shuffle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import sklearn.ensemble as ensemble
from xgboost import XGBRegressor

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

def get_days(start_dt, end_dt, on_dt, off_dt):
    if start_dt > on_dt and end_dt < off_dt:
        dt1 = [int(i) for i in start_dt.split('-')]
        dt2 = [int(i) for i in end_dt.split('-')]
    elif start_dt < on_dt and end_dt < off_dt:
        dt1 = [int(i) for i in on_dt.split('-')]
        dt2 = [int(i) for i in end_dt.split('-')]
    elif start_dt > on_dt and end_dt > off_dt:
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

def pre_this3m_train(num1, num2):
    times = [['2016-07-31', '2016-08-03', '2016-10-31'],
             ['2016-08-31', '2016-09-01', '2016-11-29'],
             ['2016-09-30', '2016-10-01', '2016-12-29'],
             ['2016-10-31', '2016-11-01', '2017-01-29'],
             ['2016-11-30', '2016-12-01', '2017-02-28'],
             ['2016-12-31', '2017-01-01', '2017-03-31'],
             ['2017-01-31', '2017-02-01', '2017-04-30']]

    t_order = pd.read_csv('./data/t_order.csv')
    t_sales_sum = pd.read_csv('./data/t_sales_sum.csv')
    t_ads = pd.read_csv('./data/t_ads.csv')
    t_comment = pd.read_csv('./data/t_comment.csv')
    t_product = pd.read_csv('./data/t_product.csv')

    data = []
    for shop_id in tqdm(range(num1, num2)):
        order_shop = t_order[t_order['shop_id'] == shop_id]
        comment_shop = t_comment[t_comment['shop_id'] == shop_id]
        product_shop = t_product[t_product['shop_id'] == shop_id]
        sales_sum_shop = t_sales_sum[t_sales_sum['shop_id'] == shop_id]
        for n, t in enumerate(times):
            order = order_shop.loc[(order_shop['ord_dt'] >= t[1]) & (order_shop['ord_dt'] <= t[2])].values
            sale_amt = offer_amt = offer_cnt = rtn_cnt = rtn_amt = ord_cnt = user_cnt = 0.0
            pids = []
            for i in range(len(order)):
                sale_amt += order[i][1]
                offer_amt += order[i][2]
                offer_cnt += order[i][3]
                rtn_cnt += order[i][5]
                rtn_amt += order[i][6]
                ord_cnt += order[i][7]
                pids.append(order[i][8])
                user_cnt += order[i][9]
            pid_num = len(list(set(pids)))

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

            ads = t_ads[t_ads['shop_id'].isin([shop_id])].values
            ads_charge = ads_consume = 0.0
            for j in range(len(ads)):
                ads_charge += ads[j][1]  # charge
                ads_consume += ads[j][2]  # consume

            product = product_shop[(product_shop['on_dt'] >= t[1]) & (product_shop['on_dt'] >= t[1])].values
            on_num = len(product)

            sales = sales_sum_shop[sales_sum_shop['dt'] == t[0]].values
            sale_sum = sales[0][2]
            data.append([shop_id, t[0], t[1], t[2], pid_num, sale_amt, offer_amt, offer_cnt, rtn_cnt, rtn_amt, ord_cnt, user_cnt, ads_charge, ads_consume, bad_rate, mid_rate, good_rate, dis_num, cmmt_num, on_num, sale_sum])
    print np.shape(data)
    columns = ['shop_id', 'dt0', 'dt1', 'dt2', 'pid_num', 'sale_amt', 'offer_amt', 'offer_cnt', 'rtn_cnt', 'rtn_amt', 'ord_cnt', 'user_cnt', 'ads_charge', 'ads_consume', 'bad_rate',
               'mid_rate', 'good_rate', 'dis_num', 'cmmt_num', 'on_num', 'sale_sum']
    df = pd.DataFrame(data, columns=columns)
    df.to_csv('./1202/this3m_%d_%d.csv' % (num1, num2-1), index=False)

def pre_this3m_test():
    times = [['2016-11-11', '2016-11-11', '2016-11-11']]

    t_order = pd.read_csv('./data/t_order.csv')
    t_ads = pd.read_csv('./data/t_ads.csv')
    t_comment = pd.read_csv('./data/t_comment.csv')
    t_product = pd.read_csv('./data/t_product.csv')

    data = []
    for shop_id in tqdm(range(1, 3001)):
        order_shop = t_order[t_order['shop_id'] == shop_id]
        comment_shop = t_comment[t_comment['shop_id'] == shop_id]
        product_shop = t_product[t_product['shop_id'] == shop_id]
        for n, t in enumerate(times):
            order = order_shop.loc[(order_shop['ord_dt'] >= t[1]) & (order_shop['ord_dt'] <= t[2])].values
            sale_amt = offer_amt = offer_cnt = rtn_cnt = rtn_amt = ord_cnt = user_cnt = 0.0
            pids = []
            for i in range(len(order)):
                sale_amt += order[i][1]
                offer_amt += order[i][2]
                offer_cnt += order[i][3]
                rtn_cnt += order[i][5]
                rtn_amt += order[i][6]
                ord_cnt += order[i][7]
                pids.append(order[i][8])
                user_cnt += order[i][9]
            pid_num = len(list(set(pids)))

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

            ads = t_ads[t_ads['shop_id'].isin([shop_id])].values
            ads_charge = ads_consume = 0.0
            for j in range(len(ads)):
                ads_charge += ads[j][1]  # charge
                ads_consume += ads[j][2]  # consume

            product = product_shop[(product_shop['on_dt'] >= t[1]) & (product_shop['on_dt'] >= t[1])].values
            on_num = len(product)

            data.append([shop_id, t[0], t[1], t[2], pid_num, sale_amt, offer_amt, offer_cnt, rtn_cnt, rtn_amt, ord_cnt, user_cnt, ads_charge, ads_consume, bad_rate, mid_rate, good_rate, dis_num, cmmt_num, on_num])
    print np.shape(data)
    columns = ['shop_id', 'dt0', 'dt1', 'dt2', 'pid_num', 'sale_amt', 'offer_amt', 'offer_cnt', 'rtn_cnt', 'rtn_amt', 'ord_cnt', 'user_cnt', 'ads_charge', 'ads_consume', 'bad_rate',
               'mid_rate', 'good_rate', 'dis_num', 'cmmt_num', 'on_num']
    df = pd.DataFrame(data, columns=columns)
    df.to_csv('./1202/this3m_test.csv', index=False)

def train(filename, num_round, early_stopping_rounds):
    data = pd.read_csv(filename).values
    np.random.shuffle(data)
    split = len(data) * 5 / 6
    X_train = data[0:split, 5:-1]
    y_train = data[0:split, -1]
    X_test = data[split:, 5:-1]
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

def predict(model, filename):
    # model = train(filename=filenames[0], num_round=1000, early_stopping_rounds=20)

    X_test1 = pd.read_csv(filename).values
    X_test1 = X_test1[:, 5:]
    test1 = xgb.DMatrix(X_test1)
    y_pred1 = model.predict(test1)
    print('y_pred1', np.shape(y_pred1))
    y_pred = []
    for i in range(3000):
        y_pred.append([i+1, y_pred1[i]]) # y_pred1[i],
    df = pd.DataFrame(y_pred)
    df.to_csv('./1202/1111_1202.csv', index=False, header=False)
    # print "The wmae is: %f" % wmae(y_test, y_pred)



def gbdt(filenames):
    data = pd.read_csv(filenames[0]).values
    np.random.shuffle(data)
    split = len(data) * 5 / 6
    X_train = data[0:split, 7:-1]
    y_train = data[0:split, -1]
    X_test = data[split:, 7:-1]
    y_test = data[split:, -1]
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42)
    # model = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.3, subsample=0.8, loss='huber', alpha=0.75,
    #                                   criterion='mse', verbose=1, random_state=20, presort='auto',) # criterion='mse',
    # model = XGBRegressor(n_estimators=1000, learning_rate=0.15, silent=False, max_depth=10, seed=50,)

    model = ensemble.ExtraTreesRegressor(n_estimators=200, criterion='mse', random_state=50, verbose=1)
    model.fit(X_train, y_train)
    # print model.feature_importances_
    preds = model.predict(X_test)
    print preds
    print np.shape(preds)
    print get_wmae(y_test, preds)
    print rmse(y_test, preds)

    X_test1 = pd.read_csv(filenames[1]).values
    X_test1 = X_test1[:, 7:]
    # test1 = xgb.DMatrix(X_test1)
    y_pred1 = model.predict(X_test1)
    print('y_pred1', np.shape(y_pred1))

    X_test2 = pd.read_csv(filenames[2]).values
    X_test2 = X_test2[:, 7:]
    # test2 = xgb.DMatrix(X_test2)
    y_pred2 = model.predict(X_test2)
    print('y_pred2', np.shape(y_pred2))

    X_test3 = pd.read_csv(filenames[3]).values
    X_test3 = X_test3[:, 7:]
    # test3 = xgb.DMatrix(X_test3)
    y_pred3 = model.predict(X_test3)
    print('y_pred3', np.shape(y_pred3))

    y_pred = []
    for i in range(3000):
        y_pred.append([i + 1, sum([y_pred1[i], y_pred2[i], y_pred3[i]]) / 3])
    df = pd.DataFrame(y_pred)
    df.to_csv('report_xgb.csv', index=False, header=False)

if __name__ == '__main__':
    start = time.time()
    # filenames = ['./1128/63000_15_new.csv', './1128/test_1.csv', './1128/test_2.csv', './1128/test_3.csv']
    # gbdt(filenames)
    # on_feature()
    # on_feature_test()

    # pre_this3m_train(1, 3001)
    # pre_this3m_test()
    model = train(filename='./1202/this3m_1_3000.csv', num_round=300, early_stopping_rounds=20)
    predict(model, filename='./1202/this3m_test.csv')
    end = time.time()
    print("Time consumption: %.2f sec." % (end - start))
