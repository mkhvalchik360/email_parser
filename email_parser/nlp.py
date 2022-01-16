import logging
import os
import regex
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
import pandas as pd
import numpy as np

from . import utils, _models_signatures
from .utils import timing
from langid.langid import LanguageIdentifier
from langid.langid import model as model_langid

# Creating language_identifier object for usage in function f_detect_language
language_identifier = LanguageIdentifier.from_modelstring(model_langid, norm_probs=True)
language_identifier.set_languages(['en', 'fr'])


logging.info(f"Reading config file from folder:{os.path.join(os.path.dirname(__file__))}")

config = utils.f_read_config(os.path.join(os.path.dirname(__file__), 'config.ini'))

device = int(config["DEFAULT"]["device"])
default_lang = config["DEFAULT"]["default_lang"]

tokenizer_dict = {}
models_dict = {}
nlp_dict = {}


dict_regex_pattern = dict(EMAIL=r'[\p{L}\p{M}\-\d._]{1,}@[\p{L}\p{M}\d\-_]{1,}(\.[\p{L}\p{M}]{1,}){1,}',
                          TEL=r'(?<!\d)(\+?\d{1,2}[ -]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}(?!\d|\p{P}\d)',
                          POST=r'\b([A-z][0-9][A-z][ -]?[0-9][A-z][0-9]|[A-z][0-9][A-z])\b',
                          PRICE=r"(([\s:,]|^){1}\$*(CA|CAD|USD|EUR|GBP|\$|\€|\£|\¢){1}\$*[\d., ]*[\d]{1,}\b)" +
                                "|([\d]{1,}[\d., ]*(CA|CAD|USD|EUR|GBP|\$|\€|\£|k|m|\¢){1,}\$*(?=\s|\p{P}|$))",
                          WEB=r"((www(\.[\p{L}\p{M}\-0-9]]{1,}){2,})" +
                              "|(https?:[^ ]*)"+
                              # r"|(([\p{L}\p{M}\.]{3,}){2,})|"
                              r"|((?<=[\s:]|^)([\p{L}\p{M}\-0-9]{1,}\.){1,}(com|ca|org|fr){1,}\b))")
                          # WEB=r"(http(s)?:\/\/)?[a-z0-9]{1}[a-z0-9-._~]+[.]{1}(com|ca)(?![\p{L}\p{M}])")

def f_load_tokenizer_and_model_for_nlp(model_name, pipeline_type='ner'):
    """
    Loading model and tokenizer takes a long time.
    We do it once and store the model and tokenizer in global dict for next usage
    Args:
        name: Name of the model that should be loaded and stored
        pipeline_type: type of pipeline that should be initialized

    Returns: tokenizer, model

    """
    global tokenizer_dict, models_dict, nlp_dict
    auto_model = None
    if pipeline_type == "ner":
        auto_model = AutoModelForTokenClassification

    if model_name not in tokenizer_dict.keys() or model_name not in models_dict.keys() or model_name not in nlp_dict.keys():
        logging.info(
            f"Loading tokenizer and model: {model_name}")
        tokenizer_dict[model_name] = AutoTokenizer.from_pretrained(model_name)
        # , add_prefix_space = True
        models_dict[model_name] = auto_model.from_pretrained(model_name)
        if pipeline_type == 'ner':
            nlp_dict[model_name] = pipeline(pipeline_type, model=models_dict[model_name], tokenizer=tokenizer_dict[model_name],
                                      aggregation_strategy="simple", device=device)


def f_ner(text, lang=default_lang):
    df_result = f_ner_regex(text)
    df_result = f_ner_model(text, lang=lang, df_result=df_result)
    return df_result


@timing
def f_ner_model(text,  lang=default_lang, df_result=pd.DataFrame()):
    list_result = []
    # We split the text by sentence and run model on each one
    sentence_tokenizer = f_split_text_by_lines(text)
    for start, end, value in sentence_tokenizer:
        if value != "":
            results = f_ner_model_by_sentence(value, lang=lang, pos_offset=start)
            if len(results) != 0:
                list_result += results
    return f_concat_results(df_result, list_result)


@timing
def f_ner_model_by_sentence(sentence, lang=default_lang, df_result=pd.DataFrame(), pos_offset=0):
    """ Run ner algorithm

    Args:
        sentence : sentence on which to run model
        lang : lang to determine which model to use
        df_result : If results of f_ner should be combined with previous value
        (in this case we will keep the previous values if tags overlapsed)

    Returns:
        Dataframe with identified entities

    """

    if not config.has_option('DEFAULT', 'ner_model_' + lang):
        raise ValueError(f"No model was defined for ner in {lang}")

    model_name = config['DEFAULT']['ner_model_' + lang]
    f_load_tokenizer_and_model_for_nlp(model_name)
    logging.debug(f"starting {model_name} on sentence:'{sentence}'")

    results = nlp_dict[model_name](sentence)
    list_result = []
    for result in results:
        if result["word"] != "" and result['entity_group'] in ["PER", "LOC", "ORG", "DATE"]:

            # Required because sometimes spaces are included in result["word"] value, but not in start/end position
            value = sentence[result["start"]:result["end"]]

            # We remove any special character at the beginning
            pattern = r"[^.,'’` \":()\n].*"
            result_regex = regex.search(pattern, value, flags=regex.IGNORECASE)

            if result_regex is not None:
                word_raw = result_regex.group()
                word = word_raw
                real_word_start = result["start"] + result_regex.start()
                real_word_end = result["start"] + result_regex.start() + len(word_raw)
                # We check if entity might be inside a longer word, if this is the case we ignore
                letter_before = sentence[max(0, real_word_start - 1): real_word_start]
                letter_after = sentence[real_word_end: min(len(sentence), real_word_end + 1)]
                if regex.match(r"[A-z]", letter_before) or regex.match(r"[A-z]", letter_after):
                    logging.debug(f"Ignoring entity {value} because letter before is"
                                  f" '{letter_before}' or letter after is '{letter_after}'")
                    continue

                list_result.append(
                    [result["entity_group"],
                     word,
                     real_word_start + pos_offset,
                     real_word_end + pos_offset,
                     result["score"]])

    return list_result


@timing
def f_concat_results(df_result, list_result_new):
    """ Merge results between existing dataframe and a list of new values

    Args:
        df_result: dataframe of entities
        list_result_new: list of new entities to be added in df_result

    Returns:
        Dataframe with all entities. Entities in list_result_new that were overlapping position of another entity in
        df_result are ignored.

    """
    # If df_result and list_result_new are both empty, we return an empty dataframe
    list_columns_names = ["entity", "value", "start", "end", "score"]
    if (df_result is None or len(df_result) == 0) and (list_result_new is None or len(list_result_new) == 0):
        return pd.DataFrame()
    elif len(list_result_new) > 0:
        if df_result is None or len(df_result) == 0:
            return pd.DataFrame(list_result_new,
                                columns=list_columns_names)
        list_row = []
        for row in list_result_new:
            df_intersect = df_result.query("({1}>=start and {0}<=end)".format(row[2], row[3]))
            if len(df_intersect) == 0:
                list_row.append(row)
        df_final = pd.concat([df_result,
                              pd.DataFrame(list_row,
                                           columns=list_columns_names)],
                             ignore_index=True) \
            .sort_values(by="start")
        return df_final
    else:
        # If list_result_new was empty we just return df_result
        return df_result


@timing
def f_detect_language(text, default=default_lang):
    """ Detect language

    Args:
        text: text on which language should be detected
        default: default value if there is an error or score of predicted value is to low (default nlp.default_lang)

    Returns:
        "fr" or "en"

    """
    lang = default
    try:
        if text.strip() != "":
            lang, score = language_identifier.classify(text.strip().replace("\n"," ").lower())
            # If scroe is not high enough we will take default value instead
            if score < 0.8:
                lang = default_lang
    except Exception as e:
        logging.error("following error occurs when trying to detect language: {}".format(e))
    finally:
        return lang

@timing
def f_find_regex_pattern(text, type_, pattern):
    """ Find all occurences of a pattern in a text and return a list of results
    Args:
        text:  the text to be analyzed
        type_:  the entity type (value is added in result)
        pattern: regex pattern to be found

    Returns:
        A list containing type, matched value, position start and end of each result

    """
    list_result = []
    results = regex.finditer(pattern, text, flags=regex.IGNORECASE)
    for match in results:
        value = match.string[match.start(): match.end()].replace("\n", " ").strip()
        list_result.append([type_,
                            value,
                            match.start(),
                            match.end(),
                            1])
    return list_result


@timing
def f_ner_regex(text, dict_pattern=dict_regex_pattern,
                df_result=pd.DataFrame()):
    """Run a series of regex expression to detect email, tel and postal codes in a full text.

    Args:
        text: the text to be analyzed
        dict_pattern: dictionary of regex expression to be ran successively (default nlp.dict_regex_pattern)
        df_result: results of this function will be merged with values provided here.
                   If value is already found at an overlapping  position in df_results, the existing value will be kept

    Returns:
        Dataframe containing results merged with provided argument df_result (if any)
    """
    logging.debug("Starting regex")
    list_result = []

    # we run f_find_regex_pattern for each pattern in dict_regex
    for type_, pattern in dict_pattern.items():
        result = f_find_regex_pattern(text, type_, pattern)
        if len(result) != 0:
            list_result += result

    df_result = f_concat_results(df_result, list_result)
    return df_result

@timing
def f_split_text_by_lines(text, position_offset=0):
    """
    :param text: text that should be split
    :return: list containing for each line:  [position start, position end, sentence]
    """
    results = []
    # iter_lines = regex.finditer(".*(?=\n|$)", text)
    iter_lines = regex.finditer("[^>\n]((.*?([!?.>] ){1,})|.*(?=\n|$))", text)
    for line_match in iter_lines:
        start_line = line_match.start()
        end_line = line_match.end()
        line = line_match.group()
        if len(line.strip()) > 1:
            results.append([start_line + position_offset, end_line + position_offset, line])
    return results


def f_detect_email_signature(text, df_ner=None, cut_off_score=0.6, lang=default_lang):
    # with tf.device("/cpu:0"):
    if text.strip() == "":
        return None
    if df_ner is None:
        df_ner = f_ner(text, lang=lang)

    df_features = _models_signatures.f_create_email_lines_features(text, df_ner=df_ner)

    if len(df_features)==0:
        return None

    #     We add dummy value for signature in order to use same function than for training of the model
    df_features["is_signature"] = -2

    x, y_out, y_mask, _, _ = _models_signatures.generate_x_y(df_features, _models_signatures.minmax_scaler,
                                                             _models_signatures.standard_scaler)

    y_predict = _models_signatures.model.predict(x)
    y_predict_value = (y_predict[y_mask != -1]> cut_off_score).reshape([-1])
    y_predict_value = np.pad(y_predict_value, (len(df_features) - len(y_predict_value), 0), constant_values=0)[
                      -len(df_features):]
    y_predict_score = y_predict[y_mask != -1].reshape([-1])
    y_predict_score = np.pad(y_predict_score, (len(df_features) - len(y_predict_score), 0), constant_values=1)[
                      -len(df_features):]

    # return(y_predict, y_mask)
    df_features["prediction"] = y_predict_value
    df_features["score"] = y_predict_score
    # return df_features
    series_position_body = df_features.query(f"""prediction==0""")['end']
    if len(series_position_body) > 0:
        body_end_pos = max(series_position_body)
    else:
        # In this case everything was detected as a signature
        body_end_pos = 0
    score = df_features.query(f"""prediction==1""")["score"].mean()
    signature_text = text[body_end_pos:].strip().replace("\n", " ")
    if signature_text != "":
        list_result = [
            # ["body", text[:body_end_pos], 0 + pos_start_email, body_end_pos + pos_start_email, 1, ""],
            ["SIGNATURE", signature_text, body_end_pos, len(text), score]]

        df_result = f_concat_results(pd.DataFrame(), list_result)
    else:
        df_result = None

    return df_result


