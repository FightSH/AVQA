import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # set gpu number
import numpy as np
import tensorflow as tf

import vggish_input
import vggish_params
import vggish_slim
import h5py
import contextlib
import wave


# get audio length
def get_audio_len(audio_file):
    # audio_file = os.path.join(audio_path, audio_name)
    with contextlib.closing(wave.open(audio_file, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        wav_length = int(frames / float(rate))
        # print("wave_len: ", wav_length)

        return wav_length


# 添加处理短音频的函数
def process_audio_with_padding(audio_file, target_length=60):
    """处理音频文件，如果长度不足则用最后1秒填充"""
    sr, snd = wavfile.read(audio_file)

    # 如果音频长度不足target_length秒
    if snd.shape[0] < sr * target_length:
        # 获取实际音频长度（秒）
        actual_length = snd.shape[0] / sr
        # 需要填充的秒数
        padding_needed = target_length - actual_length

        # 获取最后1秒的音频数据用于填充
        if snd.shape[0] > sr:  # 确保音频至少有1秒
            last_second = snd[-sr:]
        else:  # 音频不足1秒，就用全部内容
            last_second = snd

        # 计算需要重复的次数
        repeat_times = int(np.ceil(padding_needed))

        # 创建填充数据
        padding_data = np.tile(last_second, (repeat_times, 1)) if len(snd.shape) > 1 else np.tile(last_second,
                                                                                                  repeat_times)
        needed_samples = int(padding_needed * sr)
        padding_data = padding_data[:needed_samples]

        # 合并原始音频和填充数据
        padded_audio = np.vstack((snd, padding_data)) if len(snd.shape) > 1 else np.concatenate((snd, padding_data))

        return padded_audio, sr
    else:
        # 音频长度足够，不需要填充
        return snd, sr



# Paths to downloaded VGGish files.
checkpoint_path = 'vggish_model.ckpt'
pca_params_path = 'vggish_pca_params.npz'
# num_secs = 60 # length of the audio sequence. Videos in our dataset are all 10s long.
freq = 1000
sr = 44100

audio_dir = "/mnt/sda/shenhao/datasets/MUSIC-AVQA/audio/"  # .wav audio files
save_dir = "/mnt/sda/shenhao/datasets/MUSIC-AVQA/feats/qa_tiger/audit_feat/60vggish"

lis = sorted(os.listdir(audio_dir))
len_data = len(lis)
print(len_data)

i = 0

for n in range(len_data):
    i += 1

    # save file
    outfile = os.path.join(save_dir, lis[n][:-4] + '.npy')
    if os.path.exists(outfile):
        print("\nProcessing: ", i, " / ", len_data, " ----> ", lis[n][:-4] + '.npy', " is already exist! ")
        continue

    '''feature learning by VGG-net trained by audioset'''
    audio_index = os.path.join(audio_dir, lis[n])  # path of your audio files
    num_secs = 60
    num_secs_real = get_audio_len(audio_index)
    print("\nProcessing: ", i, " / ", len_data, " --------> video: ", lis[n], " ---> sec: ", num_secs_real)

    # 如果音频长度不足60秒，使用填充处理
    if num_secs_real < num_secs:
        print(f"音频长度不足{num_secs}秒，使用最后1秒填充至{num_secs}秒")
        # 使用自定义填充处理函数
        from scipy.io import wavfile

        padded_audio, sr_value = process_audio_with_padding(audio_index, num_secs)

        # 创建临时文件保存填充后的音频
        import tempfile

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_filename = temp_file.name

        # 保存填充后的音频到临时文件
        wavfile.write(temp_filename, sr_value, padded_audio.astype(np.int16))

        # 使用填充后的临时文件进行特征提取
        input_batch = vggish_input.wavfile_to_examples(temp_filename, num_secs)

        # 处理完后删除临时文件
        os.remove(temp_filename)
    else:
        # 正常处理，不需要填充
        input_batch = vggish_input.wavfile_to_examples(audio_index, num_secs_real)

    # input_batch = vggish_input.wavfile_to_examples(audio_index, num_secs)
    np.testing.assert_equal(
        input_batch.shape,
        [num_secs, vggish_params.NUM_FRAMES, vggish_params.NUM_BANDS])

    # Define VGGish, load the checkpoint, and run the batch through the model to
    # produce embeddings.
    # with tf.Graph().as_default(), tf.Session() as sess:
    with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
        vggish_slim.define_vggish_slim()
        vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)

        features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)
        [embedding_batch] = sess.run([embedding_tensor], feed_dict={features_tensor: input_batch})
        # print('VGGish embedding: ', embedding_batch[0])
        # outfile = os.path.join(save_dir, lis[n][:-4] + '.npy')
        np.save(outfile, embedding_batch)
        # audio_features[i, :, :] = embedding_batch
        print(" save info: ", lis[n][:-4] + '.npy', " ---> ", embedding_batch.shape)

        i += 1

print("\n---------------------------------- end ----------------------------------\n")


