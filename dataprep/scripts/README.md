# Scripts

- [Freiburg flatten](freiburg_flatten): Flatten the Freiburg dataset (see [Datasets > Freiburg](../datasets#freiburg))
- [Giusti flatten](giusti_flatten): Flatten the Giusti dataset (see [Datasets > Giusti](../datasets#giusti))
- [Yamaha flatten](freiburg_flatten): Unused
- [Single folder flatten](single_folder_flatten): Used to flatten our own data (like "Steentjes")
- [Remove sequential and similar images](remove_seq_sim): The giusti dataset contains a lot of similar images, this script fixes that
- [Convert masks](convert_masks): Convert the multicolor Freiburg and Yamaha masks to our own grayscale masks style
- [Images to TFR](images_to_tfr): Convert images and masks to a single [TFRecord file](https://www.tensorflow.org/tutorials/load_data/tfrecord)
