import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import os

# Read a class file and transform between class label and string
class ClassLoader(object):
    def __init__(self, class_file='./classes.txt'):
        with open(class_file, 'r') as f:
            self.classes = f.read().splitlines()
#       for i,c in enumerate(self.classes):
#           self.classes[i] = c.replace('_',' ')

        self.string2index = {c:i for i,c in enumerate(self.classes)}
        self.index2string = {i:c for i,c in enumerate(self.classes)}

    def str2ind(self, s):
        return self.string2index[s]

    def ind2str(self, i):
        return self.index2string[i]


def leafsnap_collate_fn(sample_list):
    merged = {
                'file_id': [],
                'species': [],
                'species_index': [],
                'source': [],
                'image': [],
                'segmented': []
            }
    for sample in sample_list:
        for key in sample:
            merged[key].append(sample[key])

    merged['image'] = torch.stack(merged['image'])
    merged['species_index'] = torch.LongTensor(merged['species_index'])

    return merged



class LeafSnapLoader(Dataset):
    def __init__(self,csv_file='./leafsnap-dataset-images-augmented.txt',
        leafsnap_root='./data/leafsnap', 
        source=['field'],#['lab','field'],
#       class_file='./classes.txt',
        mode='train', # train | val | test | all
        transform=None):
        self.leafsnap_root = leafsnap_root
        self.transform = transform
        self.frames = pd.read_csv(csv_file,sep='\t')
        self.frames = self.frames.loc[self.frames['source'].isin(source)]
        if mode == 'train':
            self.frames = self.frames.loc[self.frames['train']==1]
        elif mode == 'val':
            self.frames = self.frames.loc[self.frames['val']==1]
        elif mode == 'test':
            self.frames = self.frames.loc[self.frames['test']==1]
        else:
            print("Using all data")
        # If none of the above, just keep everything


#       self.classes = ClassLoader(class_file)
        
        #self.pil2tensor = transforms.ToTensor()
        
    def __len__(self):
        return len(self.frames)

    def __getitem__(self,idx):
        file_id = self.frames.iloc[idx,0]
        image_path = self.frames.iloc[idx,1]
        segmented_path = self.frames.iloc[idx,2]
        species = self.frames.iloc[idx,3]
        source = self.frames.iloc[idx,4]
        species_index = self.frames.iloc[idx,5]

        # species_index = self.classes.str2ind(species.replace(' ', '_').lower())
        # The .lower() and .replace() are necessary because in the leafsnap-dataset-images.txt file
        # the species names is capitalized and has a space instead of an underscore
        # To do: add class id to the leafsnap-dataset-images.txt file
        
        
        image = Image.open(os.path.join(self.leafsnap_root, image_path))
        segmented = Image.open(os.path.join(self.leafsnap_root, segmented_path))

        if self.transform:
            image = self.transform(image)
            segmented = self.transform(segmented)

        #image = self.pil2tensor(image)
        #segmented = self.pil2tensor(segmented)
        
        sample = {'file_id': file_id, 'species': species, 'species_index': species_index,
        'source': source, 'image': image, 'segmented': segmented}
        
        return sample


#test
if __name__ == '__main__':
    trans = transforms.Resize((224,224)) #use tranforms.Compose to add cropping etc.
    data =  LeafSnapLoader(mode='train', transform=trans)
    loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True, num_workers=4)
    print("Num batches:", len(loader))
    iterator=loader.__iter__()
    sample = iterator.next()
    print(sample['file_id'])
    print(sample['species'])
    print(sample['species_index'])
    print(sample['source'])
    print(sample['image'].shape)
    print(sample['segmented'].shape)
    images = sample['image']
    segmented = sample['segmented']
    im = images[0,:,:,:]
    seg = segmented[0,:,:,:]

    toPIL = transforms.ToPILImage()
    im = toPIL(im)
    seg = toPIL(seg)

    im.show()
    seg.show()
    
