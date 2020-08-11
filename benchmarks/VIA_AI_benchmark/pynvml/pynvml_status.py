from pynvml import *
nvmlInit()     #初始化
print("Driver: ", nvmlSystemGetDriverVersion())  #顯示驅動資訊
#>>> Driver: 384.xxx

#檢視裝置
deviceCount = nvmlDeviceGetCount()
for i in range(deviceCount):
    handle = nvmlDeviceGetHandleByIndex(i)
    print("GPU", i, ":", nvmlDeviceGetName(handle))
#>>>
#GPU 0 : b'GeForce GTX 1080 Ti'
#GPU 1 : b'GeForce GTX 1080 Ti'

#檢視視訊記憶體、溫度、風扇、電源
handle = nvmlDeviceGetHandleByIndex(0)
info = nvmlDeviceGetMemoryInfo(handle)
print("Memory Total: ",info.total)
print("Memory Free: ",info.free)
print("Memory Used: ",info.used)

print("Temperature is %d C"%nvmlDeviceGetTemperature(handle,0))
print("Fan speed is ", nvmlDeviceGetFanSpeed(handle))
print("Power ststus",nvmlDeviceGetPowerState(handle))


#最後要關閉管理工具
nvmlShutdown()
