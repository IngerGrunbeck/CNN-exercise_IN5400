import PySimpleGUI as sg
import PIL
from PIL import Image
import io
import base64
import os
import numpy as np

class GUI:
    def __init__(self, root_dir, classes, img_nr):
        self.root_dir = root_dir
        self.classes = classes
        self.img_nr = img_nr
        self.thumbnail_size = (200, 200)
        self.image_size = (800, 800)
        self.thumbnail_pad = (1, 1)
        self.screen_size = sg.Window.get_screen_size()
        self.thumbs_per_row = 5#int(self.screen_size[0]/(self.thumbnail_size[0]+self.thumbnail_pad[0])) - 1
        self.thumbs_rows = 2#int(self.screen_size[1]/(self.thumbnail_size[1]+self.thumbnail_pad[1])) - 1
        self.thumbnails_per_page = (self.thumbs_per_row, self.thumbs_rows)

    def make_square(self, im, min_size=256, fill_color=(0, 0, 0, 0)):
        x, y = im.size
        size = max(min_size, x, y)
        new_im = Image.new('RGBA', (size, size), fill_color)
        new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
        return new_im

    def convert_to_bytes(self, file_or_bytes, resize=None, fill=False):
        if isinstance(file_or_bytes, str):
            img = PIL.Image.open(file_or_bytes)
        else:
            try:
                img = PIL.Image.open(io.BytesIO(base64.b64decode(file_or_bytes)))
            except Exception as e:
                dataBytesIO = io.BytesIO(file_or_bytes)
                img = PIL.Image.open(dataBytesIO)

        cur_width, cur_height = img.size
        if resize:
            new_width, new_height = resize
            scale = min(new_height / cur_height, new_width / cur_width)
            img = img.resize((int(cur_width * scale), int(cur_height * scale)), PIL.Image.ANTIALIAS)
        if fill:
            img = self.make_square(img, self.thumbnail_size[0])
        with io.BytesIO() as bio:
            img.save(bio, format="PNG")
            del img
            return bio.getvalue()

    def display_image_window(self, filename):
        try:
            layout = [[sg.Image(data=self.convert_to_bytes(filename, self.image_size), enable_events=True)]]
            e,v = sg.Window(filename, layout, modal=True, element_padding=(0,0), margins=(0,0)).read(close=True)
        except Exception as e:
            print(f'** Display image error **', e)
            return

    def make_thumbnails(self, flist):
        layout = [[]]
        for row in range(self.thumbnails_per_page[1]):
            row_layout = []
            for col in range(self.thumbnails_per_page[0]):
                try:
                    f = flist[row*self.thumbnails_per_page[1] + col]
                    # row_layout.append(sg.B(image_data=convert_to_bytes(f, self.thumbnail_size), k=(row,col), pad=self.thumbnail_pad))
                    row_layout.append(sg.B('',k=(row,col), size=(0,0), pad=self.thumbnail_pad,))
                except:
                    pass
            layout += [row_layout]
        layout += [[sg.B(sg.SYMBOL_LEFT + ' Prev', size=(10,3), k='-PREV-'), sg.B('Next '+sg.SYMBOL_RIGHT, size=(10,3), k='-NEXT-')],
                   [sg.Combo(self.classes, enable_events=True, key='combo'),
                   sg.Button('Choose category', key='combo_choice')]
                   ]
        return sg.Window('Thumbnails', layout, element_padding=(0, 0), margins=(0, 0), finalize=True, grab_anywhere=False, location=(0,0), return_keyboard_events=True)

    EXTS = ('png', 'jpg', 'gif')

    def display_images(self, t_win, offset, files):
        currently_displaying = {}
        row = col = 0
        while True:
            if offset + 1 > len(files) or row == self.thumbnails_per_page[1]:
                break
            f = files[offset]
            currently_displaying[(row, col)] = f
            try:
                t_win[(row, col)].update(image_data=self.convert_to_bytes(f, self.thumbnail_size, True))
            except Exception as e:
                print(f'Error on file: {f}', e)
            col = (col + 1) % self.thumbnails_per_page[0]
            if col == 0:
                row += 1

            offset += 1
        if not (row == 0 and col == 0):
            while row != self.thumbnails_per_page[1]:
                t_win[(row, col)].update(image_data=sg.DEFAULT_BASE64_ICON)
                currently_displaying[(row, col)] = None
                col = (col + 1) % self.thumbnails_per_page[0]
                if col == 0:
                    row += 1

        return offset, currently_displaying

    def main(self):
        show_class = self.classes[0]
        class_file = show_class + '_filenames.npy'
        path = os.path.join(self.root_dir, class_file)
        files = np.load(path).tolist()[:self.img_nr] + np.load(path).tolist()[-self.img_nr:]
        t_win = self.make_thumbnails(files)
        offset, currently_displaying = self.display_images(t_win, 0, files)
        # offset = self.thumbnails_per_page[0] * self.thumbnails_per_page[1]
        # currently_displaying = {}
        while True:
            win, event, values = sg.read_all_windows()
            if win == sg.WIN_CLOSED:            # if all windows are closed
                break

            if event == sg.WIN_CLOSED or event == 'Exit':
                break

            if isinstance(event, tuple):
                self.display_image_window(currently_displaying.get(event))
                continue

            if event == '-NEXT-' or event.endswith('Down'):
                offset, currently_displaying = self.display_images(t_win, offset, files)
            elif event == '-PREV-' or event.endswith('Up'):
                offset -= self.thumbnails_per_page[0]*self.thumbnails_per_page[1]*2
                if offset < 0:
                    offset = 0
                offset, currently_displaying = self.display_images(t_win, offset, files)

            if event == 'combo':
                combo = values['combo']  # use the combo key

            if event == 'combo_choice':
                show_class = combo
                class_file = show_class + '_filenames.npy'
                path = os.path.join(self.root_dir, class_file)
                files = np.load(path).tolist()[:self.img_nr] + np.load(path).tolist()[-self.img_nr:]
                t_win = self.make_thumbnails(files)
                offset, currently_displaying = self.display_images(t_win, 0, files)


if __name__ == '__main__':
    gui = GUI(root_dir='../saved_scores/', classes=[
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train',
            'tvmonitor'], img_nr=10)

    gui.main()

