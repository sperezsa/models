# models

* **test_text_classif.ipynb**. Text classification (sentimental analysis) with transfer learning. Fine tuning is performed on a RoBERTa model in ES to classify reviews of a hotel in Punta Cana.
 The beltrewilton/punta-cana-spanish-reviews dataset is used, a new label is created to mark the good/bad reviews and the review is encoded with the tokenizer of the BSC-TeMU/roberta-base-bne pretrained model. 
 The model is retrained (fine-tuned) with the tokenized data and displayed in the hub [huggingface.co/Serchs/modelo](https://huggingface.co/Serchs/roberta-base-bne-finetuned-pcana_reviews-finetuned-pcana_reviews) so that it can be consulted and tested with new reviews. 
 The model is fill-mask type, it usually has better precision, instead of classic text because these usually have fixed labels and we want to catalog them like 0-bad, 1-good. 
 from transformers import AutoModelForSequenceClassification
 STATUS: Loaded and tested the model. 

* **test_gen_text.ipynb**. Text generation. It uses the model "mrm8488/spanish-gpt2" version of GPT-2 for ES based on the BETO corpus and will continue the sentence starting from the initial prompt. 
 Two decoding strategies are used, Greedy search and Beam search, and the hyperparameters of Temperature, Top-k and Top-p are used.
 STATUS: Validated.

* **test_img_clasif.ipynb**. Image Classification. The google/vit-base-patch16-224 model is used that need an image to catalog it within the 1,000 available classes.
 I recover the image from a Google Drive session, another option is from a URL. 
 STATUS: Validated.
 
* **test_traslation.ipynb**. Traslate text. It uses the Helsinki-NLP/opus-mt-en-es model to translate texts from EN to ES and need a text to translate. 
 It is also deployed in the space with Gradio: [huggingface.co/spaces/Serchs/test](https://huggingface.co/spaces/Serchs/test)
 It uses the OPUS dataset which is a large-scale parallel corpus, meaning it contains large amounts of text in multiple languages ​​aligned word by word.
 SentencePiece is a subword tokenization algorithm that splits text into smaller subunits that can be whole words, prefixes, suffixes, or individual characters.
 spm32k refers to a vocabulary containing 32,000 subunits.
 STATUS: Validated. 

* **test_load_docs_pdf.ipynb**. Using langchain, pypdf, transformers and faiss libraries to read documents and save the embeddings in a vector database to be able to make search.

* **test_Space.ipynb**. Simple example of using the Spacy library
 