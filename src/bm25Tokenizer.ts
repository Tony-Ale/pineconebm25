import * as natural from 'natural';
import * as sw from 'stopword';
import { languageToIso6393_map } from './language_map';
import * as snowball from 'snowball-stemmers'

export class BM25Tokenizer {
    public lowerCase: boolean;
    public removePunctuation: boolean;
    public removeStopwords: boolean;
    public stem: boolean;
    public language: string;
    private stemmer: snowball.Stemmer;
    private punctuation: Set<string>;

    constructor(
        lowerCase: boolean,
        removePunctuation: boolean,
        removeStopwords: boolean,
        stem: boolean,
        language: string
    ) {
        this.lowerCase = lowerCase;
        this.removePunctuation = removePunctuation;
        this.removeStopwords = removeStopwords;
        this.stem = stem;
        this.language = language;

        try{
            this.stemmer = snowball.newStemmer(this.language); // Using snowball stemmer
        }catch{
            this.stemmer = snowball.newStemmer('english')
        }

        this.punctuation = new Set([
            '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', 
            ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~'
        ]);

        if (this.stem && !this.lowerCase) {
            throw new Error(
                'Stemming applies lower case to tokens, so lower_case must be true if stem is true.'
            );
        }

    }

    public tokenize(text: string):string[] {
        let tokens = new natural.WordTokenizer().tokenize(text);

        if (this.lowerCase) {
            tokens = tokens.map(word => word.toLowerCase());
        }

        if (this.removePunctuation) {
            tokens = tokens.filter(word => !this.punctuation.has(word));
        }

        if (this.removeStopwords) {
            const stopwords = this.getStopwords()
            tokens = sw.removeStopwords(tokens, stopwords)
        }

        if (this.stem) {
            tokens = tokens.map(word => this.stemmer.stem(word));
        }

        return tokens;
    }

    getStopwords(){
        try{
            const isoLang = languageToIso6393_map[this.language]
            const stopwords = sw[isoLang as keyof typeof sw]
            if (!Array.isArray(stopwords)){
                throw new Error("Invalid stopwords")
            }
            return stopwords
        }catch{
            return sw.eng
        }
    }
}