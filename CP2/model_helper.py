import tensorflow as tf

def custom_loss(y_true, y_pred):
    # 提取出日照量的部分 (假設日照量是第4個特徵)
    sunlight_true = y_true[:, 3]
    sunlight_pred = y_pred[:, 3]
    
    # 原始均方誤差
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # 對日照量部分的誤差進行加權
    sunlight_mse = tf.reduce_mean(tf.square(sunlight_true - sunlight_pred))
    
    # 將日照量的影響放大
    weighted_loss = mse + 1.2 * sunlight_mse  # 將日照量誤差放大1.2倍
    return weighted_loss