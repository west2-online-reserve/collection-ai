import numpy as np
import onnxruntime as ort
from torchvision import transforms
from PIL import Image
import time


class NPUInference:
    def __init__(self, model_path):
        """
        初始化NPU推理器
        Args:
            model_path: ONNX模型路径
        """
        # 检查可用的执行提供程序
        available_providers = ort.get_available_providers()
        print("可用的执行提供程序:", available_providers)

        # 配置会话选项
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # 创建推理会话，优先使用DirectML（AMD NPU）
        providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers
        )

        # 获取输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        print(f"模型输入形状: {input_shape}")

        # 准备图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, image_path):
        """
        预处理输入图像
        Args:
            image_path: 图像路径
        Returns:
            预处理后的图像张量
        """
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image.unsqueeze(0).numpy()  # 添加batch维度

    def inference(self, image_path):
        """
        执行推理
        Args:
            image_path: 输入图像路径
        Returns:
            推理结果
        """
        # 预处理图像
        input_data = self.preprocess_image(image_path)

        # 记录推理时间
        start_time = time.time()

        # 执行推理
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_data}
        )

        end_time = time.time()
        inference_time = (end_time - start_time) * 1000  # 转换为毫秒

        return outputs[0], inference_time


def main():
    # 使用示例
    # 假设你有一个ONNX格式的模型文件
    model_path = "your_model.onnx"  # 替换为你的模型路径
    image_path = "test_image.jpg"   # 替换为你的测试图像路径

    try:
        # 初始化推理器
        print("初始化NPU推理器...")
        inferencer = NPUInference(model_path)

        # 执行推理
        print("执行推理...")
        results, inference_time = inferencer.inference(image_path)

        # 输出结果
        print(f"\n推理完成！")
        print(f"推理时间: {inference_time:.2f} ms")
        print(f"输出形状: {results.shape}")

        # 如果是分类任务，可以添加后处理
        # probabilities = softmax(results[0])
        # predicted_class = np.argmax(probabilities)

    except Exception as e:
        print(f"发生错误: {str(e)}")
        print("\n请确保：")
        print("1. 已安装正确的AMD驱动程序")
        print("2. 已安装onnxruntime-directml")
        print("3. 模型文件路径正确")
        print("4. 输入图像存在")


if __name__ == "__main__":
    main()
