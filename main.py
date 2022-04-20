import cv2
import torch
from time import time
import models as mdl
import numpy as np
import inspect
import albumentations as al
from albumentations.pytorch import ToTensorV2
from music import MusicGesture
import vlc


lst = ['right FIST',
       'left FIST',
       'right PALM',
       'left PALM',
       'right 2FINGER',
       'left 2FINGER',
       'right L',
       'left L']


def get_model_transforms(arch, xp):
    model = dict(inspect.getmembers(mdl))[arch](8, 3)
    model.load_state_dict(torch.load(rf'logging/XP{xp}_model'))
    model.eval()
    data_transform_t = al.Compose([
        al.Resize(120, 160, always_apply=True),
        al.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(transpose_mask=True)
    ])
    return model, data_transform_t


def main(music, model, data_transforms, sensitivity=5, confidence=.8):
    vid = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_PLAIN
    prev = 0
    frames = np.empty((sensitivity, 1))
    frames.fill(np.nan)
    fps = 12
    try:
        while True:
            ret, frame = vid.read()
            prev = time() - prev
            if time() - prev > 1/fps:
                frame_t = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_t = data_transforms(image=frame_t)['image']
                with torch.no_grad():
                    out = model(frame_t.unsqueeze(0)).softmax(-1)
                    predict = out.argmax(axis=-1).item() if out.max().item() > confidence else np.nan
                    frames = np.append(frames[1:], predict)
            print(frames)
            command = lst[int(frames[-1])] if np.all(frames[-1] == frames) else 'undetected'
            music.gesture(command)
            text = f'{command}, confidence: {out.max().item()}' if command != 'undetected' else command
            cv2.putText(frame,
                        text,
                        (50, 50),
                        font, 1,
                        (0, 0, 255),
                        2,
                        cv2.LINE_4)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                music.player.stop()
                break
    finally:
        vid.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    ml = vlc.MediaList()
    ml.add_media('/music_samples/test1.mp3')
    ml.add_media('/music_samples/test2.mp3')
    ml.add_media('/music_samples/test3.mp3')
    mp = vlc.MediaListPlayer()
    mp.set_media_list(ml)
    player = MusicGesture(mp, None)
    mdl, data_t = get_model_transforms('Var7', 32)
    main(player, mdl, data_t, confidence=.85)
