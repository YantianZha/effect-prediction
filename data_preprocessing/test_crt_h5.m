dset = single(rand(5,10,3));
dset_details.Location = bomkdir('./dataset');

dset_details.Name = 'Random';

attr = 'Some random data';
attr_details.Name = 'Description';
attr_details.AttachedTo = '/group1/dataset1.2/Random';
attr_details.AttachType = 'dataset';

hdf5write('myfile2.h5', dset_details, dset, attr_details, attr, ...
    'WriteMode', 'overwrite');