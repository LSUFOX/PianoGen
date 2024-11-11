from hw1_yh import Composer
import torch
from midi2seq import process_midi_seq, seq2piano
from torch.utils.data import DataLoader, TensorDataset

def main():
    # 定义批处理大小
    bsz = 32  # 你可以根据需要调整大小
    epoch = 2  # 设置训练的轮次

    # 控制 load_trained 参数，决定是否加载预训练模型
    load_trained = False  # 设置为 True 时加载预训练模型，False 时重新训练

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载MIDI序列数据
    print("Starting to process MIDI files...")
    piano_seq = torch.from_numpy(process_midi_seq(datadir='.')).to(device)  # 将数据加载到设备上
    print("Finished processing MIDI files.")

    # 创建数据加载器
    print("Loading data into DataLoader...")
    loader = DataLoader(TensorDataset(piano_seq), shuffle=True, batch_size=bsz, num_workers=0)
    print("Data loaded successfully.")

    # 初始化Composer对象，根据load_trained决定是否加载预训练模型
    #cps = Composer(load_trained=load_trained).to(device)  # 将模型加载到指定设备
    cps = Composer(load_trained=load_trained)
    # 训练模型
    if not load_trained:  # 如果从头训练模型
        print("Starting training...")
        for i in range(epoch):
            for x in loader:
                cps.train(x[0].cuda(0).long())  # 调用模型的 train 方法
        print("Training finished.")
    else:
        print("Using pre-trained model for composition.")

    # 生成音乐
    midi = cps.compose(100)  # 将生成的序列移动到 CPU
    midi = seq2piano(midi)
    midi.write('piano1.midi')
    print("Generated piano1.midi.")

if __name__ == "__main__":
    main()
