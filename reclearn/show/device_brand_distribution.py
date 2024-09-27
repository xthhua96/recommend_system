import pandas as pd
import matplotlib.pyplot as plt

# 读取数据文件
file_path = "../data/app_request_brand_number(oneday).txt"
data = pd.read_csv(file_path, sep="\t")

# 提取品牌和数量
brands = data.iloc[:, 0]
quantities = data.iloc[:, 1]

# 对品牌按数量进行排序，获取前10大品牌
top_brands = data.nlargest(10, columns=[quantities.name])  # 取前10大品牌及其数量
other_quantity = quantities[
    ~brands.isin(top_brands[brands.name])
].sum()  # 合并其余品牌数量

# 更新品牌和数量，前10大品牌+“其他”
brands = top_brands[brands.name].tolist() + ["Others"]
quantities = top_brands[quantities.name].tolist() + [other_quantity]

# 绘制饼状图
plt.figure(figsize=(10, 6))
plt.pie(quantities, labels=brands, autopct="%1.1f%%", startangle=140)
plt.title("Device Brand Distribution")
plt.axis("equal")  # 确保饼图是圆形
plt.savefig("../result/device_brand_distribution.png")
plt.close()
