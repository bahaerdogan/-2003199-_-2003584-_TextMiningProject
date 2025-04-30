import pandas as pd
import re
import os
import csv

custom_stopwords = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
    "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
    'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself',
    'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
    'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
    'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
    'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
    'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
    'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
    'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
    'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
    'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're',
    've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn',
    "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma',
    'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't",
    'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't",
    'wouldn', "wouldn't"
}

slang_dict = {
    "afaik": "as far as i know",
    "afk": "away from keyboard",
    "asap": "as soon as possible",
    "atm": "at the moment",
    "btw": "by the way",
    "cu": "see you",
    "faq": "frequently asked questions",
    "fyi": "for your information",
    "gtg": "got to go",
    "idk": "i do not know",
    "imo": "in my opinion",
    "imho": "in my humble opinion",
    "lol": "laughing out loud",
    "omg": "oh my god",
    "rofl": "rolling on the floor laughing",
    "tbh": "to be honest",
    "thx": "thanks",
    "tq": "thank you",
    "ttyl": "talk to you later",
    "u": "you",
    "ur": "your",
    "y": "why",
    "yolo": "you only live once",
    "brb": "be right back",
    "np": "no problem",
    "gr8": "great",
    "thx": "thanks",
    "pls": "please",
    "bc": "because",
    "cuz": "because",
    "b4": "before",
    "2day": "today",
    "tmrw": "tomorrow",
    "gn": "good night",
    "gm": "good morning",
    "dm": "direct message",
    
    "fav": "favorite",
    "fn": "fantastic",
    "mins": "minutes",
    "hrs": "hours",
    "cgi": "computer generated imagery",
    "fx": "effects",
    "sfx": "special effects",
    "vfx": "visual effects",
    "3d": "three dimensional",
    "pic": "picture",
    "flick": "film",
    "rom com": "romantic comedy",
    "seq": "sequel",
    "preq": "prequel",
    "tril": "trilogy",
    "dir": "director",
    "prod": "production",
    "chars": "characters",
    "char": "character",
    "def": "definitely",
    "prob": "probably",
    "fwd": "forward",
    "pov": "point of view",
    "ott": "over the top",
    "oscar": "academy award",
    "noms": "nominations",
    "nom": "nomination",
    "perf": "performance",
    "cam": "camera",
    "rec": "recommend",
    "rewatchable": "worth watching again",
    "binge": "watch continuously",
    "masterp": "masterpiece",
    "5star": "five star",
    "10of10": "ten out of ten",
    "mindblown": "amazed",
    "storyline": "plot",
    "orig": "original",
    "bg": "background",
    "soundtrk": "soundtrack",
    "mvp": "most valuable performer",
    "spoiler": "plot reveal",
    "meh": "mediocre",
    "overrated": "rated too highly",
    "underrated": "not rated highly enough",
    "disappointing": "not meeting expectations",
    "visuals": "visual effects",
    "cinemato": "cinematography",
    "imax": "image maximum",
    "theatre": "theater",
    "tkt": "ticket",
    
    "awsm": "awesome",
    "gud": "good", 
    "amazn": "amazing",
    "incrdbl": "incredible",
    "fab": "fabulous",
    "superb": "excellent",
    "bril": "brilliant",
    "terribl": "terrible",
    "horribl": "horrible",
    "awfl": "awful",
    "mstr pce": "masterpiece",
    "sux": "sucks",
    "wste": "waste",
    "wtf": "what the heck",
    "wtch": "watch",
    "wld": "would",
    "cld": "could",
    "shld": "should",
    "n": "and",
    "w": "with",
    "w/o": "without",
    "b": "be",
    "bcm": "become",
    "bcz": "because",
    "abt": "about",
    "dat": "that",
    "dis": "this",
    "luv": "love",
    "hte": "hate",
    "eva": "ever",
    "neva": "never",
    "evry": "every",
    "da": "the"
}

def reduce_repeated_characters(word):
    return re.sub(r'(.)\1{2,}', r'\1\1', word)

def replace_slang_words(word):
    return slang_dict.get(word, word)

def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)

    tokens = text.split()
    reduced_tokens = [reduce_repeated_characters(word) for word in tokens]
    slang_replaced = [replace_slang_words(word) for word in reduced_tokens]
    
    cleaned_text = ' '.join(slang_replaced)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    final_tokens = [word for word in cleaned_text.split() if word not in custom_stopwords]

    return ' '.join(final_tokens)

def read_raw_file(file_path):
    lines = []
    encodings = ['utf-8', 'latin1', 'cp1252']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                lines = f.readlines()
            print(f"Dosya '{encoding}' kodlaması ile başarıyla okundu.")
            break
        except UnicodeDecodeError:
            print(f"{encoding} kodlaması ile okuma başarısız.")
            continue
    
    if not lines:
        raise Exception("Dosya hiçbir kodlama ile okunamadı.")
    
    first_line = lines[0].strip()
    separators = [',', ';', '\t', '|']
    detected_sep = None
    
    for sep in separators:
        if sep in first_line:
            detected_sep = sep
            field_count = first_line.count(sep) + 1
            print(f"Olası ayraç: '{sep}', Alan sayısı: {field_count}")
    
    if not detected_sep:
        print("Bilinen ayraç bulunamadı, tek sütun olarak işlem yapılacak.")
        data = [{'text': line.strip()} for line in lines]
        return pd.DataFrame(data)
    
    headers = lines[0].strip().split(detected_sep)
    
    data = []
    for line_num, line in enumerate(lines[1:], 2):
        try:
            fields = line.strip().split(detected_sep)
            row = {}
            
            if len(fields) != len(headers):
                print(f"Uyarı: Satır {line_num}, alan sayısı ({len(fields)}) başlık sayısına ({len(headers)}) eşit değil.")
                while len(fields) < len(headers):
                    fields.append("")
                if len(fields) > len(headers):
                    extra = " ".join(fields[len(headers)-1:])
                    fields = fields[:len(headers)-1] + [extra]
            
            for i, header in enumerate(headers):
                row[header] = fields[i] if i < len(fields) else ""
            
            data.append(row)
        except Exception as e:
            print(f"Satır {line_num} okunurken hata: {str(e)}")
    
    return pd.DataFrame(data)

def main():
    input_file = 'movie_reviews_with_sentiment_normalized.xlsx - movie_reviews_with_sentiment_no.csv'
    output_file = 'normalized_output_extra.csv'
    text_column = 'review'
    
    if not os.path.exists(input_file):
        print(f"Hata: '{input_file}' dosyası bulunamadı.")
        return
    
    try:
        try:
            print("Standart CSV okuyucu deneniyor...")
            df = pd.read_csv(input_file)
            print(f"CSV başarıyla okundu. Toplam {len(df)} satır, {len(df.columns)} sütun.")
        except Exception as e:
            print(f"Standart CSV okuma hatası: {str(e)}")
            try:
                print("Python CSV motoru deneniyor...")
                df = pd.read_csv(input_file, sep=None, engine='python', on_bad_lines='warn')
                print(f"CSV başarıyla okundu. Toplam {len(df)} satır, {len(df.columns)} sütun.")
            except Exception as e2:
                print(f"Python CSV motoru hatası: {str(e2)}")
                try:
                    print("Farklı quote karakterleri deneniyor...")
                    df = pd.read_csv(input_file, quoting=csv.QUOTE_NONE, escapechar='\\', encoding='utf-8')
                    print(f"CSV başarıyla okundu. Toplam {len(df)} satır, {len(df.columns)} sütun.")
                except Exception as e3:
                    print(f"Quote ayarları ile okuma hatası: {str(e3)}")
                    print("Manuel dosya okuma yöntemi deneniyor...")
                    df = read_raw_file(input_file)
                    print(f"Dosya manuel olarak okundu. Toplam {len(df)} satır, {len(df.columns)} sütun.")
        
        print(f"Sütunlar: {', '.join(df.columns)}")
        
        if text_column not in df.columns:
            if len(df.columns) == 1:
                text_column = df.columns[0]
                print(f"'{text_column}' sütunu kullanılıyor.")
            else:
                print(f"Hata: '{text_column}' adlı sütun bulunamadı.")
                text_column = input("Normalizasyon için sütun adını girin (mevcut sütunlar yukarıda listelenmiştir): ")
                if text_column not in df.columns:
                    print(f"Hata: '{text_column}' adlı sütun bulunamadı.")
                    return
        
        print("\nİlk 5 satır:")
        print(df[[text_column]].head())
        
        print("\nNormalizasyon işlemi uygulanıyor...")
        df['clean_text'] = df[text_column].apply(normalize_text)
        
        empty_count = df['clean_text'].apply(lambda x: x == '').sum()
        if empty_count > 0:
            print(f"Bilgi: Normalizasyon sonrası {empty_count} satır boş metin içeriyor.")
            
        df.to_csv(output_file, index=False)
        print(f"Normalizasyon tamamlandı. Çıktı dosyası: {output_file}")
        
        original_chars = df[text_column].fillna('').astype(str).str.len().sum()
        normalized_chars = df['clean_text'].str.len().sum()
        reduction_percent = 100 * (1 - normalized_chars / original_chars) if original_chars > 0 else 0
        
        print(f"Toplam karakter sayısı: {original_chars:,} → {normalized_chars:,}")
        print(f"Metin boyutu azalma oranı: %{reduction_percent:.2f}")
        
        if 'sentiment' in df.columns:
            sentiment_counts = df['sentiment'].value_counts()
            print("\nSentiment Dağılımı:")
            for sentiment, count in sentiment_counts.items():
                print(f"{sentiment}: {count} yorum ({count/len(df)*100:.1f}%)")
        
        word_counts = {}
        for text in df['clean_text']:
            if isinstance(text, str):
                for word in text.split():
                    if len(word) > 2:
                        word_counts[word] = word_counts.get(word, 0) + 1
        
        top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        print("\nEn sık kullanılan 20 kelime (normalizasyon sonrası):")
        for word, count in top_words:
            print(f"{word}: {count}")
            
        print("\nNormalizasyon öncesi ve sonrası örnek:")
        for i in range(min(5, len(df))):
            print(f"\nÖrnek {i+1}:")
            print(f"Film: {df['film'].iloc[i]}")
            print(f"Sort Type: {df['sort_type'].iloc[i]}")
            print(f"Sentiment: {df['sentiment'].iloc[i]}")
            print(f"Orijinal: {df[text_column].iloc[i]}")
            print(f"Normalize: {df['clean_text'].iloc[i]}")
        
    except Exception as e:
        print(f"Beklenmeyen hata: {str(e)}")

if __name__ == "__main__":
    main()