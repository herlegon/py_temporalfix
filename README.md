# py_temporalfix
A Python wrapper for running the ["TemporalFix Classic"](https://github.com/pifroggi/vs_temporalfix#temporalfix-classic) script developed by [pifroggi](https://github.com/pifroggi).
_This project is experimental_
&nbsp;

## Table of Contents
- [Basic usage](#Basic-usage)
- [Installation](#Installation)
- [Usage](#Usage)
- [Troubleshooting](#Troubleshooting)

&nbsp;


## Basic usage
```bash
python py_temporalfix.py --input input_video.mkv --output output_video.mkv --t_radius 6 --strength 300
```

&nbsp;

_Note: The output path is optional. If omitted, a suffix will automatically be appended to the input filename._

&nbsp;

## Installation
- **Requirements:** Windows 11 and Python 3.14
> [!TIP]
> Using [Miniconda](https://docs.anaconda.com/free/miniconda/index.html) is recommended to create an isolated Python environment rather than using your system-wide Python installation.

- (Optional) **Create a conda environment**
    ```bash
    conda create -n m python==3.14
    conda activate m
    ```

- **Clone or download the repository**
Clone the repository: `git clone https://github.com/herlegon/py_temporalfix`
Alternatively, download and extract the repository as a ZIP archive.


- **Install dependencies and external tools**
  Install the required Python packages and download a portable VapourShit Python environment (`vspython`) to the `external` directory:
    ```bash
    pip install -r requirements.txt
    python install.py
    ```

&nbsp;

## Usage

```bash
python py_temporalfix.py --help
```

### Input, output
| Option&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| Description |
| :--- | :--- |
| `--input` | Path to the input video file. |
| `--output`| Path to the output video file. |
| `--suffix`| Suffix to append to the output filename when `--output` is not specified. Defaults to `_<t_radius>_<strength>`)|


### Parameters Passed to TemporalFix Classic
For more details: ["TemporalFix Classic"](https://github.com/pifroggi/vs_temporalfix#temporalfix-classic)

| Option&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |  Default | Description |
| :--- | :---: | :--- |
| `--t_radius` | `6` | Sets the temporal radius, i.e. the number of neighboring frames considered for temporal averaging. Higher values generally produce more stable results. However, performance drops significantly when `t_radius` exceeds 6. |
| `--strength` | `400` | Controls the strength of temporal inconsistency suppression. Higher values result in more aggressive suppression. Reduce this value if you notice blending, ghosting around small movements, or blocky artifacts. |


## Troubleshooting

Add `--log` to the command line to enable diagnostic output. The script will:
- Parse the script and display diagnostic information.
- Display the encoder command line.
- Generate a log file in the output directory.



&nbsp;


## Not yet supported:


### Video encoding
Parameters passed to the encoder. Refer to the [FFmpeg documentation](https://ffmpeg.org/ffmpeg-all.html)

| Option&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |   Default  | Description
| :--- | :---: | :--- |
| `--vcodec`    | `h264`    |  `h264`, `h265`, `ffv1`, `vp9` |
| `--pix_fmt`   | `yuv420p` |  rgb or yuv formats  |
| `--preset`    |  -         | `ultrafast`, `superfast`, `veryfast`, `faster`, `fast`, `medium`, `slow`, `slower`, `veryslow`   |
| `--crf`       |    `23`|  0 to 51  |
| `--tune`      |  `film`, `animation`, `grain`, `stillimage`, `fastdecode`, `zerolatency`  |
| `--ffmpeg_args` | `""` | Used to pass customized arguments to the encoder (FFmpeg). Override the previous options. This option must be double quoted. Example: `--ffmpeg_args "-preset veryfast"`|

| Argument  | Format        | Description           |
| :--- | :---: | :--- |
| `-ss`     | hh:mm:ss.ms   | seek start            |
| `-t`      | hh:mm:ss.ms   | duration              |
| `-to`     | hh:mm:ss.ms   | position              |

&nbsp;
