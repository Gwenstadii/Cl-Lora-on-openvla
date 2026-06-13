# vla-scripts/check_rlds_keys.py
import tensorflow_datasets as tfds
import tensorflow as tf

def inspect_rlds():
    # 关闭 GPU，防止抢占显存
    tf.config.set_visible_devices([], 'GPU')
    
    dataset_name = "libero_spatial_no_noops"
    data_dir = "/root/autodl-tmp/modified_libero_rlds"
    
    print(f"Loading dataset {dataset_name} from {data_dir}...")
    builder = tfds.builder(dataset_name, data_dir=data_dir)
    dataset = builder.as_dataset(split='train')
    
    # 取出第一个 episode (一条完整轨迹)
    for episode in dataset.take(1):
        print("\n========== Episode 级别键名 ==========")
        print(list(episode.keys()))
        
        # RLDS 中 steps 也是一个嵌套结构，我们取出这根轨迹的第一帧 (Step)
        for step in episode['steps'].take(1):
            print("\n========== 单帧 (Step) 级别键名与形状 ==========")
            for key, value in step.items():
                if isinstance(value, dict):
                    print(f"[{key}] (嵌套字典):")
                    for sub_key, sub_value in value.items():
                        print(f"  ├── '{sub_key}': shape={sub_value.shape}, dtype={sub_value.dtype}")
                else:
                    print(f"[{key}]: shape={value.shape}, dtype={value.dtype}")
        break  # 看完第一帧就退出

if __name__ == "__main__":
    inspect_rlds()