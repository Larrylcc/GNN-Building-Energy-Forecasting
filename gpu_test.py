import lightgbm as lgb
print(lgb.__version__)
import lightgbm as lgb
import numpy as np

X = np.random.rand(1000, 10)
y = np.random.rand(1000)

params = {
    "objective": "regression",
    "device_type": "gpu",
    "gpu_platform_id": 0,
    "gpu_device_id": 0
}

model = lgb.LGBMRegressor(**params, n_estimators=10)

model.fit(X, y)

print("training finished")

import pyopencl as cl

for platform in cl.get_platforms():
    print("Platform:", platform.name)
    for device in platform.get_devices():
        print("  Device:", device.name)