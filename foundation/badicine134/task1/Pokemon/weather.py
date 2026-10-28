import random

class Weather:
    def __init__(self) -> None:
        self.current_weather: str = "Normal"
        self.duration: int = 3

    def update_weather(self):
        #减少天气持续时间
        if self.duration > 0:
            self.duration -= 1

            if self.duration == 0 and self.current_weather != "Normal":
                self.current_weather = "Normal"
                self.duration = 2
                print('\n天气恢复了正常。')
                return

            if self.duration == 0 and self.current_weather == "Normal":
                self.random_weather()

    def weather_announce(self):
        if self.current_weather == "Sunny":
            print(f"\n 大晴天！(持续 {self.duration} 回合)")
            print(" -> 效果：火属性技能威力提升，水属性特性性能下降，草属性特性恢复效果增强。")
        elif self.current_weather == "Rain":
            print(f"\n 暴雨！(持续 {self.duration} 回合)")
            print(" -> 效果：水属性特性性能提高，火属性技能威力下降。")
        elif self.current_weather == "Thunderstorm":
            print(f"\n 雷暴！(持续 {self.duration} 回合)")
            print(" -> 效果：电属性技能威力提升，闪避率提高。")
        else:
            print('天气恢复正常')

    # 覆盖当前天气
    def cover_weather(self, weather: str, duration: int):
        self.current_weather = weather
        self.duration = duration
        # 播报天气
        self.weather_announce()

    def random_weather(self):
        # 随机生成天气
        weather_list = ["Sunny", "Rain", "Thunderstorm"]
        self.current_weather = random.choice(weather_list)
        self.duration = random.randint(2, 4)
        # 播报天气
        print("【随机生成天气】！")
        self.weather_announce()

    def get_weather_mult(self,skill_type):
        #根据天气补正各种效果
        if self.current_weather == "Sunny":
            if skill_type == "Fire":
                return 1.3
            if skill_type == "Water":
                return 0.7
        if self.current_weather == "Rain":
            if skill_type == "Water":
                return 1.3
            if skill_type == "Fire":
                return 0.7
        if self.current_weather == "Thunderstorm":
            if skill_type == "Electric":
                return 1.3
        return 1.0