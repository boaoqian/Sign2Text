import click
import tensorflow as tf
import V2T, Model
import os
import warnings
warnings.filterwarnings('ignore')



@click.group()
def cli():
    if len(tf.config.list_physical_devices()) < 2:
        click.echo('Warning: GPU not found, use cpu instead')


@click.command()
@click.option('-m', default='base', help='available models:base,large')
@click.option('-f', default='cli', help='output format:cli,csv,txt')
@click.option('-i', help='input file or directory for batch processing')
def translate(m, f, i):
    raw_result = []
    input_file = []
    model = Model.Model(m)
    if os.path.isfile(i):
        input_file.append(i)
    else:
        input_file = os.listdir(i)

    for file in input_file:
        data = V2T.video2text(i+'/'+file)
        raw_result.append(model.predict(data))

    if f == 'cli':
        print("output:")
        for i in range(len(raw_result)):
            print(f"{input_file[i]}: {raw_result[i]}")

    elif f == 'txt':
        with open(i + 'result.txt', 'w') as f:
            for i in range(len(raw_result)):
                f.write(f"{input_file[i]}: {raw_result[i]}\n")

    elif f == "csv":
        with open(i + 'result.csv', 'w') as f:
            for i in range(len(raw_result)):
                f.write(f"{input_file[i]},{raw_result[i]}\n")

@click.command()
@click.option('-i', help='Directory where training data is stored')
def dataset(i):
    data_path = os.listdir(i)
    video_path = [i for i in data_path if i.split('.')[-1] != 'txt']
    y_dict = dict()
    with open(i+'/'+'label.txt', 'r') as f:
        for line in f:
            name, y = line.split(',')
            y_dict[name] = y[:-1].encode()

    with tf.io.TFRecordWriter("dataset.tfrecords") as file_writer:
        for file in video_path:
            x = V2T.video2text(i+'/'+file)
            y = y_dict[file.split('/')[-1]]
            x = tf.convert_to_tensor(x)
            x = tf.io.serialize_tensor(x)
            record_bytes = tf.train.Example(features=tf.train.Features(feature={
                "x": tf.train.Feature(bytes_list=tf.train.BytesList(value=[x.numpy()])),
                "y": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y])),
            })).SerializeToString()
            file_writer.write(record_bytes)


cli.add_command(translate)
cli.add_command(dataset)
if __name__ == '__main__':
    cli()
