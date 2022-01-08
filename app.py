import logging, regex
import gradio
from email_parser import utils, nlp
from email_parser.doc_email import Email

def print_highlighted_text(text, df_result, offset=0):
    iter_match = regex.finditer("\s|$", text)
    start_pos = 0
    list_values = []
    for match in iter_match:
        word = match.string[start_pos:match.start()]

        df_entity = df_result.query(f"{start_pos + offset}>=start & {match.start() + offset}<=end").head(1)
        if len(df_entity) == 1:
            entity = df_entity["entity"].values[0]
        else:
            entity = None
        list_values.append((word, entity))
        # list_values.append((match.string[match.start():match.end()], None))
        start_pos = match.end()
    return list_values


def display_email(text, part=1):
    doc = Email(text)
    list_emails = doc.list_emails
    if part <= len(list_emails):
        text = list_emails[int(part-1)]["body"]
        header = list_emails[int(part-1)]["header"]
        lang = nlp.f_detect_language(text)

        if len(header)>0:
            df_results_header = nlp.f_ner(header, lang=lang)
            df_results_header = Email.f_find_person_in_header(header, df_result=df_results_header)
            list_words_headers = print_highlighted_text(header, df_results_header)
        else:
            list_words_headers = None

        df_result = nlp.f_ner(text, lang=lang)
        logging.debug(f"NER results for text '{text}' are: {df_result}")
        df_signature = nlp.f_detect_email_signature(text, df_ner=df_result)
        if df_signature is not None and len(df_signature) > 0:
            start_signature_position = df_signature["start"].values[0]
            text_body = text[:start_signature_position]
            text_signature = text[start_signature_position:]
            list_words_signature = print_highlighted_text(text_signature, df_result, offset=start_signature_position)
        else:
            text_body = text
            list_words_signature = None
        list_words_body = print_highlighted_text(text_body, df_result)

        return None, lang, list_words_headers, list_words_body, list_words_signature
    else:
        return f"Email number {int(part)} was requested but only {len(list_emails)} emails was found in this thread", \
               None, None, None, None


utils.f_setup_logger(level_sysout=logging.ERROR, level_file=logging.DEBUG, folder_path="logs")


iface = gradio.Interface(title="Parser of email",
                         description="Small application that can extract a specific email in a thread of email,"
                                     " highlights the entities found in the text (person, organization, date,...)"
                                     " and extract email signature if any.",
                         fn=display_email,
                         inputs=["textbox",
                             gradio.inputs.Number(default=1, label="Email number in thread")],
                         outputs=[
                              gradio.outputs.Textbox(type="str", label="Error"),
                              gradio.outputs.Textbox(type="str", label="Language"),
                              gradio.outputs.HighlightedText(label="Header"),
                              gradio.outputs.HighlightedText(label="Body"),
                              gradio.outputs.HighlightedText(label="Signature")],
                        examples=[["""Bonjour Vincent,
Merci de m’avoir rappelé hier.
Seriez vous disponible pour un rendez vous la semaine prochaine?
Merci,
Jean-Baptiste""", 1],  ["""Hello Jack,

I hope you had nice holiday as well.
Please find attached the requested documents,

Best Regards,
George
Vice president of Something
email: george@google.com
tel: 512-222-5555

On Mon, Jan 7, 2022 at 12:39 PM, Jack <jack@google.com> wrote:

Hello George,

I wish you a happy new year. I hope you had nice holidays.
Did you see Garry during your vacation?
Do you have the documents I requested earlier?

Thanks,
Jack


""", 1] ,  ["""Hello Jack,

I hope you had nice holiday as well.
Please find attached the requested documents,

Best Regards,
George
Vice president of Something
email: george@google.com
tel: 512-222-5555

On Mon, Jan 7, 2022 at 12:39 PM, Jack <jack@google.com> wrote:

Hello George,

I wish you a happy new year. I hope you had nice holidays.
Did you see Garry during your vacation?
Do you have the documents I requested earlier?

Thanks,
Jack


""", 2] ])


iface.launch()