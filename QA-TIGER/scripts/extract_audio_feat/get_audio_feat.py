import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # set gpu number
import numpy as np
import tensorflow as tf

import vggish_input
import vggish_params
import vggish_slim
# import h5py # h5py seems unused in the provided snippet, can be removed if not needed elsewhere
import contextlib
import wave
from scipy.io import wavfile # Added for reading/writing wav for padding
import tempfile # Added for temporary padded files


# get audio length
def get_audio_len(audio_file):
    # audio_file = os.path.join(audio_path, audio_name)
    with contextlib.closing(wave.open(audio_file, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        wav_length = int(frames / float(rate))
        # print("wave_len: ", wav_length)

        return wav_length


# Function to generate padded audio data
def generate_padded_audio_data(audio_file_path, target_length_secs):
    """
    Reads an audio file. If its duration is less than target_length_secs,
    it pads the audio data using its last second.
    Returns the audio data (as numpy array) and its sample rate.
    If audio is already long enough, returns original data and sample rate.
    """
    sr, snd = wavfile.read(audio_file_path)

    if not isinstance(snd, np.ndarray):
        # This case should ideally not happen if wavfile.read works as expected
        raise TypeError(f"wavfile.read did not return a NumPy array for {audio_file_path}")

    actual_samples = snd.shape[0]
    target_samples = int(target_length_secs * sr)

    if actual_samples < target_samples:
        base_name = os.path.basename(audio_file_path)
        print(f"音频 {base_name} (实际: {actual_samples/sr:.2f}s) 时长不足 {target_length_secs}s，将使用最后一秒数据进行填充。")

        if actual_samples == 0:
            print(f"警告: 音频文件 {base_name} 为空。将生成 {target_length_secs}s 静音。")
            dtype_to_use = snd.dtype if hasattr(snd, 'dtype') and snd.size > 0 else np.int16
            num_channels = snd.shape[1] if len(snd.shape) > 1 and snd.shape[1] > 0 else 1
            if num_channels > 1:
                 padded_audio_data = np.zeros((target_samples, num_channels), dtype=dtype_to_use)
            else:
                 padded_audio_data = np.zeros(target_samples, dtype=dtype_to_use)
            return padded_audio_data, sr

        samples_in_one_second = sr
        if actual_samples >= samples_in_one_second:
            padding_source_data = snd[-samples_in_one_second:]
        else:
            padding_source_data = snd

        if padding_source_data.shape[0] == 0:
            print(f"警告: 无法提取用于填充的源数据 {base_name} (源数据为空)。将使用静音填充剩余部分。")
            remaining_samples_to_pad = target_samples - actual_samples
            dtype_to_use = snd.dtype
            num_channels = snd.shape[1] if len(snd.shape) > 1 else 1
            silence_shape = (remaining_samples_to_pad, num_channels) if num_channels > 1 else (remaining_samples_to_pad,)
            silence_padding = np.zeros(silence_shape, dtype=dtype_to_use)
            padded_audio_data = np.concatenate((snd, silence_padding), axis=0)
            return padded_audio_data, sr

        padding_needed_samples = target_samples - actual_samples

        num_repeats_needed = int(np.ceil(padding_needed_samples / padding_source_data.shape[0]))

        if len(padding_source_data.shape) > 1: # Multi-channel
            tiled_padding_block = np.tile(padding_source_data, (num_repeats_needed, 1))
        else: # Mono
            tiled_padding_block = np.tile(padding_source_data, num_repeats_needed)

        full_padding_data = tiled_padding_block[:padding_needed_samples]

        padded_audio_data = np.concatenate((snd, full_padding_data), axis=0)

        return padded_audio_data, sr
    else:
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
short_audio_files = []

for n in range(len_data):
    i += 1

    outfile = os.path.join(save_dir, lis[n][:-4] + '.npy')
    if os.path.exists(outfile):
        print("\nProcessing: ", i, " / ", len_data, " ----> ", lis[n][:-4] + '.npy', " is already exist! ")
        continue

    audio_index = os.path.join(audio_dir, lis[n])
    num_secs = 60
    num_secs_real = get_audio_len(audio_index)

    current_audio_input_path = None
    temp_audio_path_for_vggish = None # To store path of temporary file

    try:
        if num_secs_real < num_secs:
            short_audio_files.append(lis[n])
            audio_data_to_process, sample_rate_of_data = generate_padded_audio_data(audio_index, num_secs)

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_f:
                temp_audio_path_for_vggish = tmp_f.name
            wavfile.write(temp_audio_path_for_vggish, sample_rate_of_data, audio_data_to_process.astype(np.int16))
            current_audio_input_path = temp_audio_path_for_vggish
        else:
            current_audio_input_path = audio_index

        print("\nProcessing: ", i, " / ", len_data, " --------> video: ", lis[n], " ---> sec: ", num_secs_real, f"(using input: {os.path.basename(current_audio_input_path)})")

        input_batch = vggish_input.wavfile_to_examples(current_audio_input_path, num_secs)
        np.testing.assert_equal(
            input_batch.shape,
            [num_secs, vggish_params.NUM_FRAMES, vggish_params.NUM_BANDS])

        with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
            vggish_slim.define_vggish_slim()
            vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)

            features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
            embedding_tensor = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)
            [embedding_batch] = sess.run([embedding_tensor], feed_dict={features_tensor: input_batch})
            np.save(outfile, embedding_batch)
            print(" save info: ", lis[n][:-4] + '.npy', " ---> ", embedding_batch.shape)

    except Exception as e:
        print(f"Error processing file {lis[n]} (input: {current_audio_input_path}): {e}")
        # Decide if you want to continue to the next file or stop
    finally:
        if temp_audio_path_for_vggish and os.path.exists(temp_audio_path_for_vggish):
            try:
                os.remove(temp_audio_path_for_vggish)
            except Exception as e_del:
                print(f"Warning: Could not delete temporary file {temp_audio_path_for_vggish}: {e_del}")

print("\n---------------------------------- end ----------------------------------\n")

if short_audio_files:
    print("以下音频文件时长不足60秒：")
    for audio_name in short_audio_files:
        print(audio_name)
else:
    print("没有发现时长不足60秒的音频文件。")

