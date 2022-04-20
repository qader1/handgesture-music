from time import time


class MusicGesture:
    def __init__(self, player, lst):
        self.player = player
        self.list = lst
        self.last_change = None
        self.last_vol = None
        self.vol_time = .15
        self.change_time = .8
        self.vol = 100
        self.mapping = {
            'right FIST': 'pause',
            'left FIST': 'unpause',
            'right PALM': 'stop',
            'left PALM': 'play',
            'right 2FINGER': 'next',
            'left 2FINGER': 'previous',
            'right L': 'up',
            'left L': 'down',
            'undetected': None
        }

    def gesture(self, gest):
        command = self.mapping[gest]
        if command == 'pause' and not self.player.is_playing():
            self.player.pause()
        elif command == 'unpause' and self.player.is_playing():
            self.player.pause()
        elif command == 'play':
            self.player.play()
        elif command == 'stop':
            self.player.stop()
        elif command in ['next', 'previous']:
            if self.last_change is None or time() - self.last_change > self.change_time:
                if command == 'next':
                    self.player.next()
                else:
                    self.player.previous()
                self.last_change = time()
        elif command in ['up', 'down']:
            if self.last_vol is None or time() - self.last_vol > self.vol_time:
                if command == 'up':
                    if self.vol >= 92:
                        self.vol = 100
                    else:
                        self.vol += 2
                elif command == 'down':
                    if self.vol <= 2:
                        self.vol = 0
                    else:
                        self.vol -= 2
                self.player.get_media_player().audio_set_volume(self.vol)







