"""
TensorFlow/Keras 版本的損失函數
從 PyTorch 版本轉換而來
"""

import tensorflow as tf


def mse_loss_for_variable_length_data():
    """
    建立用於可變長度資料的 MSE 損失函數
    
    Returns:
        loss_function: 接受 (ipt, target, n_frames_list) 的損失函數
    """
    
    def loss_function(ipt, target, n_frames_list=None):
        """
        計算可變長度資料的 MSE 損失
        
        Args:
            ipt: 預測輸出 [B, F, T] 或 [B, F, T, 1]
            target: 目標 [B, F, T] 或 [B, F, T, 1]
            n_frames_list: 每個樣本的實際幀數列表（可選）
            
        Returns:
            MSE 損失值
        """
        E = 1e-8
        
        # 如果是 4D tensor，squeeze 最後一維
        if len(ipt.shape) == 4:
            ipt = tf.squeeze(ipt, axis=-1)
        if len(target.shape) == 4:
            target = tf.squeeze(target, axis=-1)
        
        # 自動裁切時間軸對齊
        min_T = tf.minimum(tf.shape(target)[-1], tf.shape(ipt)[-1])
        target = target[..., :min_T]
        ipt = ipt[..., :min_T]
        
        # 如果沒有提供 n_frames_list，使用簡單 MSE
        if n_frames_list is None:
            return tf.reduce_mean(tf.square(ipt - target))
        
        # 若是 batch=1，無需 mask
        batch_size = tf.shape(target)[0]
        if batch_size == 1:
            return tf.reduce_mean(tf.square(ipt - target))
        
        # 建立 mask
        # target shape: [B, F, T]
        freq_bins = tf.shape(target)[1]
        max_frames = tf.shape(target)[2]
        
        # 建立二進位 mask [B, T]
        frame_indices = tf.range(max_frames, dtype=tf.int32)
        n_frames_tensor = tf.cast(n_frames_list, tf.int32)
        
        # [B, T] mask
        time_mask = tf.less(
            tf.expand_dims(frame_indices, 0),  # [1, T]
            tf.expand_dims(n_frames_tensor, 1)  # [B, 1]
        )
        time_mask = tf.cast(time_mask, tf.float32)  # [B, T]
        
        # 擴展到 [B, F, T]
        binary_mask = tf.expand_dims(time_mask, 1)  # [B, 1, T]
        binary_mask = tf.tile(binary_mask, [1, freq_bins, 1])  # [B, F, T]
        
        # 應用 mask
        masked_ipt = ipt * binary_mask
        masked_target = target * binary_mask
        
        # 計算 MSE
        loss = tf.reduce_sum(tf.square(masked_ipt - masked_target))
        loss = loss / (tf.reduce_sum(binary_mask) + E)
        
        return loss
    
    return loss_function


class MSELossVariableLength(tf.keras.losses.Loss):
    """
    Keras Loss 類別版本，用於可變長度資料的 MSE
    
    使用方式:
        loss_fn = MSELossVariableLength()
        model.compile(loss=loss_fn)
    """
    
    def __init__(self, name="mse_variable_length"):
        super().__init__(name=name)
        
    def call(self, y_true, y_pred):
        """
        計算損失
        
        注意：這個版本不支援 n_frames_list，
        如果需要可變長度支援，請使用 mse_loss_for_variable_length_data() 函數
        """
        # 自動裁切時間軸對齊
        min_T = tf.minimum(tf.shape(y_true)[-1], tf.shape(y_pred)[-1])
        y_true = y_true[..., :min_T]
        y_pred = y_pred[..., :min_T]
        
        return tf.reduce_mean(tf.square(y_pred - y_true))


# =====================================================================
# 測試程式碼
# =====================================================================

if __name__ == "__main__":
    print("測試損失函數")
    
    # 建立測試資料
    batch_size = 4
    freq_bins = 51
    max_time = 200
    
    ipt = tf.random.normal([batch_size, freq_bins, max_time])
    target = tf.random.normal([batch_size, freq_bins, max_time])
    n_frames_list = [200, 180, 150, 100]  # 每個樣本的實際長度
    
    # 測試函數版本
    loss_fn = mse_loss_for_variable_length_data()
    
    # 無 mask 的損失
    loss_no_mask = loss_fn(ipt, target, n_frames_list=None)
    print(f"Loss (no mask): {loss_no_mask:.6f}")
    
    # 有 mask 的損失
    loss_with_mask = loss_fn(ipt, target, n_frames_list=n_frames_list)
    print(f"Loss (with mask): {loss_with_mask:.6f}")
    
    # 測試類別版本
    loss_class = MSELossVariableLength()
    loss_class_result = loss_class(target, ipt)
    print(f"Loss (class): {loss_class_result:.6f}")
    
    print("\n✅ 損失函數測試成功！")

