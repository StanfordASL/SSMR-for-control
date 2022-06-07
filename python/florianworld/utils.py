import argparse
import logging
import os
import re
from typing import Dict, Optional

import numpy as np
import pandas as pd
import distutils.util


class ArgparseKeyValueAction(argparse.Action):
    # Constructor calling
    def __call__(self, parser, namespace,
                 values, option_string=None):
        setattr(namespace, self.dest, dict())

        for value in values:
            key, value = value.split('=')
            value = str_to_likely_type(value)
            getattr(namespace, self.dest)[key] = value


def str_to_likely_type(value: str):
    try:
        value = int(value)
    except ValueError:
        try:
            value = float(value)
        except ValueError:
            try:
                value = distutils.util.strtobool(value) == 1
            except ValueError:
                pass
                # try to parse array
                if value.startswith('[') and value.endswith(']'):
                    try:
                        value = eval(value)
                    except:
                        pass
    return value


def get_nans_in_trajectory(t_xyz_wxyz):
    traj_has_nans = np.any(np.isnan(t_xyz_wxyz), axis=1)
    nan_percentage = np.count_nonzero(traj_has_nans) / len(traj_has_nans) * 100
    return nan_percentage, traj_has_nans


def timestamp_to_rosbag_time(timestamps, df_rt):
    return np.interp(timestamps, df_rt['ts_real'], df_rt['t_sim'])


def timestamp_to_rosbag_time_zero(timestamps, df_rt):
    return timestamp_to_rosbag_time(timestamps, df_rt) - df_rt['t_sim'][0]


def timestamp_to_real_time(timestamps, df_rt):
    return np.interp(timestamps, df_rt['ts_real'], df_rt['t_real'])


def name_to_identifier(name):
    return name.lower().replace(' ', '_')


def convert_eklt_to_rpg_tracks(df_tracks, file=None):
    tracks = df_tracks.loc[df_tracks.update_type != 'Lost', ['id', 'patch_t_current', 'center_x',
                                                             'center_y']].to_numpy()
    if file is not None:
        np.savetxt(file, tracks)
    return tracks


def convert_xvio_to_rpg_tracks(df_tracks, file=None):
    tracks = df_tracks[['id', 't', 'x_dist', 'y_dist']].to_numpy()
    if file is not None:
        np.savetxt(file, tracks)
    return tracks


# def read_json_file(output_folder):
#     profile_json_filename = os.path.join(output_folder, "profiling.json")
#     if os.path.exists(profile_json_filename):
#         with open(profile_json_filename, "rb") as f:
#             profiling_json = orjson.loads(f.read())
#     else:
#         profiling_json = None
#     return profiling_json


def read_output_files(output_folder, gt_available):
    df_poses = pd.read_csv(os.path.join(output_folder, "pose.csv"), delimiter=";")
    df_features = pd.read_csv(os.path.join(output_folder, "features.csv"), delimiter=";")
    df_resources = pd.read_csv(os.path.join(output_folder, "resource.csv"), delimiter=";")
    df_groundtruth = None
    if gt_available:
        df_groundtruth = pd.read_csv(os.path.join(output_folder, "gt.csv"), delimiter=";")
    df_realtime = pd.read_csv(os.path.join(output_folder, "realtime.csv"), delimiter=";")

    df_xvio_tracks = pd.read_csv(os.path.join(output_folder, "xvio_tracks.csv"), delimiter=";")

    df_imu_bias = None
    imu_bias_filename = os.path.join(output_folder, "imu_bias.csv")
    if os.path.exists(imu_bias_filename):
        df_imu_bias = pd.read_csv(imu_bias_filename, delimiter=";")


    df_ekf_updates = None
    ekf_updates_filename = os.path.join(output_folder, "ekf_updates.csv")
    if os.path.exists(ekf_updates_filename):
        df_ekf_updates = pd.read_csv(ekf_updates_filename, delimiter=";")

    # profiling_json = read_json_file(output_folder)
    return df_groundtruth, df_poses, df_realtime, df_features, df_resources, df_xvio_tracks, df_imu_bias, df_ekf_updates


def read_eklt_output_files(output_folder):
    df_events = pd.read_csv(os.path.join(output_folder, "events.csv"), delimiter=";")
    df_optimizations = pd.read_csv(os.path.join(output_folder, "optimizations.csv"), delimiter=";")
    df_eklt_tracks = pd.read_csv(os.path.join(output_folder, "eklt_tracks.csv"), delimiter=";")
    return df_events, df_optimizations, df_eklt_tracks


def rms(data):
    return np.linalg.norm(data) / np.sqrt(len(data))


def nanrms(data):
    without_nans = data[~np.isnan(data)]
    return np.linalg.norm(without_nans) / np.sqrt(len(without_nans))


def n_to_grid_size(n):
    cols = 1
    rows = 1
    while n > cols * rows:
        if cols - rows < 2:  # this number should adapt, but works fine up to ~30
            cols = cols + 1
        else:
            cols = cols - 1
            rows = rows + 1
    return rows, cols


class DynamicAttributes:
    def __getattr__(self, item):
        if item not in self.__dict__.keys():
            return None
        return self.__dict__[item]

    def __setattr__(self, key, value):
        self.__dict__[key] = value


def merge_tables(tables, column=None):
    result_table = None
    for t in tables:
        if result_table is None:
            result_table = t
        else:
            if column:
                result_table = pd.merge(result_table, t, on=column)
            else:
                result_table = pd.merge(result_table, t, left_index=True, right_index=True)
    return result_table


def get_quantized_statistics_along_axis(x, data, data_filter=None, resolution=0.1):
    # TODO: this code fails if min(x) == max(x)
    buckets = np.arange(np.min(x), np.max(x), resolution)
    bucket_index = np.digitize(x, buckets)
    indices = np.unique(bucket_index)

    # filter empty buckets:  (-1 to convert upper bound --> lower bound, as we always take the first errors per bucket)
    buckets = buckets[np.clip(indices - 1, 0, len(buckets))]

    stats_func = get_common_stats_functions()

    stats = {x: np.empty((len(indices))) for x in stats_func.keys()}

    for i, idx in enumerate(indices):
        data_slice = data[bucket_index == idx]

        if data_filter:
            data_slice = data_filter(data_slice)

        for k, v in stats.items():
            stats[k][i] = stats_func[k](data_slice)

    return buckets, stats


def get_common_stats_functions():
    stats_func = {
        'mean': lambda d: np.mean(d),
        'median': lambda d: np.median(d),
        'min': lambda d: np.min(d),
        'max': lambda d: np.max(d),
        'q25': lambda d: np.quantile(d, 0.25),
        'q75': lambda d: np.quantile(d, 0.75),
        'q05': lambda d: np.quantile(d, 0.05),
        'q95': lambda d: np.quantile(d, 0.95),
        'num': lambda d: len(d)
    }
    return stats_func


def read_neurobem_trajectory(filename):
    input_traj = pd.read_csv(filename)
    # zero align
    input_traj['t'] -= input_traj['t'][0]
    t_xyz_wxyz = input_traj[["t", "pos x", "pos y", "pos z", "quat w", "quat x", "quat y", "quat z"]].to_numpy()
    trajectory = PoseTrajectory3D(t_xyz_wxyz[:, 1:4], t_xyz_wxyz[:, 4:8], t_xyz_wxyz[:, 0])
    return trajectory


def read_x_evaluate_gt_csv(gt_csv_filename):
    df_groundtruth = pd.read_csv(gt_csv_filename, delimiter=";")
    evo_trajectory, _ = convert_to_evo_trajectory(df_groundtruth)
    return evo_trajectory


def read_esim_trajectory_csv(csv_filename):
    df_trajectory = pd.read_csv(csv_filename)
    # columns: ['# timestamp', ' x', ' y', ' z', ' qx', ' qy', ' qz', ' qw']

    df_trajectory['# timestamp'] /= 1e9

    t_xyz_wxyz = df_trajectory[['# timestamp', ' x', ' y', ' z', ' qw', ' qx', ' qy', ' qz']].to_numpy()
    return convert_t_xyz_wxyz_to_evo_trajectory(t_xyz_wxyz)


def get_ros_topic_name_from_msg_type(input_bag, msg_type: str):
    topic_info = input_bag.get_type_and_topic_info()
    event_topics = [k for k, t in topic_info.topics.items() if t.msg_type == msg_type]
    if len(event_topics) > 1:
        logging.warning("multiple event topics found (%s), taking first: '%s'", event_topics, event_topics[0])
    elif len(event_topics) == 0:
        raise LookupError("No dvs_msgs/EventArray found in bag")
    event_topic = event_topics[0]
    return event_topic


def read_all_ros_msgs_from_topic_into_dict(event_topic, input_bag):
    event_array_messages = {}
    for topic, msg, t in input_bag.read_messages([event_topic]):
        if t in event_array_messages:
            logging.warning("Multiple messages at time %s in topic %s", t, topic)
        event_array_messages[t.to_sec()] = msg
    return event_array_messages


def prepare_output_folder(output_folder: Optional[str], fall_back: str):
    if output_folder is None:
        output_folder = fall_back
    output_folder = os.path.normpath(output_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    return output_folder


def convert_matlab_to_numpy(input_file_name, output_dir):
    in_function = False
    indents = "    "

    output_file_name = os.path.join(output_dir, os.path.basename(input_file_name)[:-2] + ".py")

    # the following subpatterns are used:
    #  - variable names / function names: [a-zA-Z_][a-zA-Z_0-9]*
    #  - doubles: ([+-]?(?:[\d]+\.?|[\d]*\.[\d]+))(?:[Ee][+-]?[\d]+)?

    regex_function = r"^function ([a-zA-Z_][a-zA-Z_0-9]*) = ([a-zA-Z_][a-zA-Z_0-9]*)\(([a-zA-Z_0-9 ,]+)\)"
    regex_comment = r"^%(.*)"
    regex_semicolon_not_at_end = r"([^;]);([^;])"
    regex_trailing_semicolon = r"(.*);$"
    regex_math_function = r"([^a])(cos|sin|tan|sqrt)\("
    regex_math_function_arc = r"(a(cos|sin|tan))\("
    regex_math_op = r"\.([\*\/])"
    regex_power = r"(\(.*\)|[a-zA-Z_][a-zA-Z_0-9]*|([+-]?(?:[\d]+\.?|[\d]*\.[\d]+))(?:[Ee][+-]?[\d]+)?)" \
                  r"[\.]?\^(\(.*\)|([+-]?(?:[\d]+\.?|[\d]*\.[\d]+))(?:[Ee][+-]?[\d]+)?)"
    regex_array_build_up = r" = (\[.*\])$"
    regex_array_access = r"\((\d+),:\)"  # Only match [number, :] type of array access
    regex_struct_access = r"\{(\d+)\}"
    regex_struct_access_general = r"\{(.+)\}"
    regex_reshape = r"reshape\(\[(.*)\],\[?(\d+),(\d+)\]?\)"
    regex_end = r"^end$"

    compiled_regex_function = re.compile(regex_function)
    compiled_regex_comment = re.compile(regex_comment)
    compiled_regex_semicolon_not_at_end = re.compile(regex_semicolon_not_at_end)
    compiled_regex_trailing_semicolon = re.compile(regex_trailing_semicolon)
    compiled_regex_math_function = re.compile(regex_math_function)
    compiled_regex_math_function_arc = re.compile(regex_math_function_arc)
    compiled_regex_math_op = re.compile(regex_math_op)
    compiled_regex_power = re.compile(regex_power)
    compiled_regex_array_build_up = re.compile(regex_array_build_up)
    compiled_regex_array_access = re.compile(regex_array_access)
    compiled_regex_struct_access = re.compile(regex_struct_access)
    compiled_regex_struct_access_general = re.compile(regex_struct_access_general)
    compiled_regex_reshape = re.compile(regex_reshape)
    compiled_regex_end = re.compile(regex_end)

    return_value = None

    with open(output_file_name, "w") as output_file:

        output_file.write("import numpy as np\n\n")

        with open(input_file_name) as input_file:
            for line in input_file:
                match_function = compiled_regex_function.match(line)

                if match_function:
                    return_value = match_function.group(1)
                    function_name = match_function.group(2)
                    parameters = match_function.group(3)
                    output_file.write(F"def {function_name}({parameters}):\n")

                    in_function = True
                else:
                    line = compiled_regex_comment.sub(r"#\g<1>", line)
                    line = compiled_regex_trailing_semicolon.sub(r"\g<1>", line)
                    line = compiled_regex_math_function.sub(r"\g<1>np.\g<2>(", line)
                    line = compiled_regex_math_function_arc.sub(r"np.arc\g<2>(", line)
                    line = compiled_regex_math_op.sub(r"\g<1>", line)
                    line = compiled_regex_power.sub(r"np.power(\g<1>, \g<3>)", line)
                    # this works for now but makes not a lot of sense
                    line = compiled_regex_reshape.sub(r"np.reshape(np.hstack([\g<1>]), (\g<3>, \g<2>)).T", line)
                    line = compiled_regex_array_build_up.sub(r" = np.array(\g<1>)", line)
                    line = compiled_regex_semicolon_not_at_end.sub("\g<1>,\g<2>", line)

                    array_access = compiled_regex_array_access.search(line)
                    if array_access:
                        line = compiled_regex_array_access.sub(F"[{int(array_access.group(1))-1}]", line)

                    struct_access = compiled_regex_struct_access.search(line)
                    if struct_access:
                        line = compiled_regex_struct_access.sub(F"[{int(struct_access.group(1)) - 1}]", line)

                    # do this for remaining {}
                    line = compiled_regex_struct_access_general.sub(r"[\g<1>]", line)

                    end_match = compiled_regex_end.match(line)
                    if in_function:
                        if end_match:
                            output_file.write(F"{indents}return {return_value}\n")
                            in_function = False
                        else:
                            output_file.write(F"{indents}{line}")
                    else:
                        output_file.write(F"{line}")

        if in_function:
            output_file.write(F"{indents}return {return_value}\n")
