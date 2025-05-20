import numpy as np

class lithium_ion_battery():
    # working voltage 3.8
    # cut-off voltage 2
    # we have 100 cells
    def __init__(self, capacity):
        self.cell_voltage = 4
        self.cutoff_voltage = 2
        self.grade = (self.cell_voltage - self.cutoff_voltage) / 100
        self.total_capacity = capacity
        self.capacity = capacity * 0.9  # 初始90%电量
        self.Ah = capacity / (self.cell_voltage * 100)
        self.need_charge = False
        self.energy_consume = 0
        self.SOC = 0.9

    def use(self, duration, power):
        # 先消耗能量
        self.energy_consume = duration * power / (3600)
        self.capacity -= self.energy_consume

        # 重新计算SOC
        self.SOC = self.capacity / self.total_capacity
        if self.SOC > 0.9:
            self.SOC = 0.9
        if self.SOC <= 0.2:
            self.need_charge = True
        else:
            self.need_charge = False

        return self.need_charge

    def charge(self, wh):
        self.capacity = min(self.capacity + wh, self.total_capacity)
        self.SOC = self.capacity / self.total_capacity
        if self.SOC > 0.9:
            self.SOC = 0.9
            self.capacity = self.total_capacity * 0.9
        self.need_charge = False
