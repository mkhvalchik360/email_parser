import logging
import pandas as pd
import numpy as np
import regex
import os
import configparser
from sentence_transformers import SentenceTransformer
from scipy.spatial import distance
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from tensorflow import keras
import pickle

from . import nlp, utils

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), 'config.ini'))



model_name = config["DEFAULT"]["name_model_signature"]

model = keras.models.load_model(filepath=utils.get_model_full_path(model_name))
minmax_scaler = pickle.load(open(utils.get_model_full_path(model_name +"/minmax_scaler.p"), "rb"))
standard_scaler = pickle.load(open(utils.get_model_full_path(model_name +"/standard_scaler.p"), "rb"))


list_name_columns_features = ["line_number",
                              "text",
                              "start",
                              "end",
                              "PER", "ORG", "LOC", "DATE", "TEL", "EMAIL", "WEB",
                              "SIGNATURE",
                              "word_count",
                              "inv_distance_to_merci",
                              "inv_distance_to_cordlt",
                              "inv_distance_to_regards",
                              "inv_distance_to_sincerely",
                              "inv_distance_to_sent_from",
                              "start_with_ps", "position_line",
                              "special_characters_count", "empty_chars_with_prev_line"]

list_columns_used_in_model = ["PER", "ORG", "LOC", "DATE", "TEL", "EMAIL",
                              # "WEB",
                              "word_count",
                              "inv_distance_to_merci",
                              "inv_distance_to_cordlt",
                              # "inv_distance_to_regards",
                              "inv_distance_to_sincerely",
                              "inv_distance_to_sent_from",
                              "start_with_ps",
                              "position_line",
                              "special_characters_count",
                              "empty_chars_with_prev_line"]

columns_to_scale_minmax = ["PER", "ORG", "LOC", "DATE", "TEL", "EMAIL", "WEB", "position_line",
                           "empty_chars_with_prev_line",
                           "inv_distance_to_merci",
                           "inv_distance_to_cordlt",
                           "inv_distance_to_regards",
                           "inv_distance_to_sincerely",
                           "inv_distance_to_sent_from",
                           "start_with_ps"
                           ]

columns_to_scale_standard = ["word_count", "special_characters_count"]

def f_retrieve_entities_for_line(df_ner, start=0, end=1e12):
    """Retrieve all entities in the previously computed dataframe  for a specific line

    Args:
          df_ner:  dataframe containing found entities
          start:  start position of the line in original text
          end: end position of the line in original text

          """

    if len(df_ner) > 0:
        df = df_ner.query(f"""(start>= {start}  and end <= {end}) or (start<={start}  and end>={end})""")
        return df


embedder_model = SentenceTransformer("distiluse-base-multilingual-cased-v1")


def f_create_embedding_inv_dist_feature(text1, text2):
    """ Computing distance between two texts based on their embedding
    provided by the SentenceTransformer above"""
    embedding_merci = embedder_model.encode(text1)
    embedding_line = embedder_model.encode(text2)
    dist = distance.cosine(embedding_merci, embedding_line)
    return min(5, 1 / (dist + 0.0001))


def f_create_email_lines_features(text, df_ner=None, position_offset=0):
    list_lines = nlp.f_split_text_by_lines(text, position_offset)
    list_features_vectors = []
    if df_ner is None:
        df_ner = nlp.f_ner(text)

    for line_number in range(0, len(list_lines)):
        list_features_vectors.append(f_create_line_features(list_lines, line_number, df_ner))

    df_features = pd.DataFrame(list_features_vectors, columns=list_name_columns_features)

    return df_features



def f_create_line_features(list_lines, line_number, df_ner):
    current_line = list_lines[line_number]
    total_lines = len(list_lines)
    features_vector = [line_number, current_line[2], current_line[0], current_line[1]]
    logging.debug(f"Creating line features for {current_line}")
    df_ner_line = f_retrieve_entities_for_line(df_ner=df_ner, start=current_line[0], end=current_line[1])

    # Adding entity to feature vector
    for entity in ["PER", "ORG", "LOC", "DATE", "TEL", "EMAIL", "WEB", "SIGNATURE"]:
        value = len(df_ner_line.query(f"entity=='{entity}'")) if df_ner_line is not None else 0
        features_vector.append(value)
    # Adding word count
    features_vector.append(len(current_line[2].split()))
    # distance to greeting word "merci"
    features_vector.append(f_create_embedding_inv_dist_feature("merci", current_line[2].lower()))

    # distance to greeting word "merci"
    features_vector.append(f_create_embedding_inv_dist_feature("cordialement", current_line[2].lower()))

    # distance to greeting word "regards"
    features_vector.append(f_create_embedding_inv_dist_feature("regards", current_line[2].lower()))

    # distance to greeting word "regards"
    features_vector.append(f_create_embedding_inv_dist_feature("sincerely", current_line[2].lower()))

    # distance to  word "sent from"
    features_vector.append(f_create_embedding_inv_dist_feature("sent from", current_line[2].lower()))

    # Line start with ps:
    features_vector.append(regex.match(r"\s*ps *:", current_line[2],  flags=regex.IGNORECASE ) is not None)

    # Adding position line in email
    position_in_email = (line_number + 1) / total_lines
    features_vector.append(position_in_email)
    # Adding special character count
    special_char_count = len(regex.findall(r"[^\p{L}0-9 .,\n]", current_line[2]))
    features_vector.append(special_char_count)
    # Number of empty chars with previous line
    empty_chars_with_prev_line = 0 if line_number == 0 else current_line[0] - list_lines[line_number - 1][1]
    features_vector.append(empty_chars_with_prev_line)
    return features_vector


def generate_x_y(df, minmax_scaler=None, standard_scaler=None, n_last_lines_to_keep=30,
                 list_columns=list_columns_used_in_model):
    df, minmax_scaler, standard_scaler = f_scale_parameters(df, minmax_scaler, standard_scaler)
    x = df[list_columns].to_numpy()[-n_last_lines_to_keep:, :]
    x = np.expand_dims(x, axis=0)
    x = pad_sequences(x, dtype='float64', value=0, maxlen=n_last_lines_to_keep)

    y = df["is_signature"].to_numpy()[-n_last_lines_to_keep:]
    y = np.expand_dims(y, axis=0)
    y_out = pad_sequences(y, value=0, maxlen=n_last_lines_to_keep)
    y_mask = pad_sequences(y,  value=-1, maxlen=n_last_lines_to_keep)
    return x, y_out, y_mask, minmax_scaler, standard_scaler

def f_scale_parameters(df_tagged_data, minmax_scaler=None, standard_scaler=None):
    # df_tagged_data = df_tagged_data.copy(deep=True)
    if minmax_scaler is None:
        logging.debug("fitting new min max scaller")
        minmax_scaler = MinMaxScaler()
        df_tagged_data.loc[:, columns_to_scale_minmax] = minmax_scaler.fit_transform(
            df_tagged_data[columns_to_scale_minmax])
    else:
        logging.debug("using already fitted minmax scaler")
        df_tagged_data.loc[:, columns_to_scale_minmax] = minmax_scaler.transform(
            df_tagged_data[columns_to_scale_minmax])

    if standard_scaler is None:
        logging.debug("fitting new standard scaler")
        standard_scaler = StandardScaler()
        df_tagged_data.loc[:, columns_to_scale_standard] = standard_scaler.fit_transform(
            df_tagged_data[columns_to_scale_standard])
    else:
        logging.debug("using already fitted scaler")
        df_tagged_data.loc[:, columns_to_scale_standard] = standard_scaler.transform(
            df_tagged_data[columns_to_scale_standard])
    return df_tagged_data, minmax_scaler, standard_scaler