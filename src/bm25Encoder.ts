import { BM25Tokenizer } from "./bm25Tokenizer";
import * as murmurhash from "murmurhash-js";
import fs from "fs/promises";
import path from "path";
import fetch from "node-fetch";

// Define interfaces for SparseVector
export interface SparseVector {
  indices: number[];
  values: number[];
}

// BM25Encoder class
export class BM25Encoder {
  private b: number;
  private k1: number;
  private _tokenizer: BM25Tokenizer;
  private doc_freq: { [key: number]: number } | null = null;
  private n_docs: number | null = null;
  private avgdl: number | null = null;

  constructor(
    b: number = 0.75,
    k1: number = 1.2,
    lower_case: boolean = true,
    remove_punctuation: boolean = true,
    remove_stopwords: boolean = true,
    stem: boolean = true,
    language: string = "english"
  ) {
    this.b = b;
    this.k1 = k1;
    this._tokenizer = new BM25Tokenizer(
      lower_case,
      remove_punctuation,
      remove_stopwords,
      stem,
      language
    );
  }

  // Function to check if a file exists
  static async fileExists(filePath: string): Promise<boolean> {
    try {
      await fs.access(filePath);
      return true;
    } catch {
      return false;
    }
  }

  // Fit BM25 to a corpus
  fit(corpus: string[]): this {
    // TODO is .every efficient? or i should just check per iteration in the loop
    if (
      !Array.isArray(corpus) ||
      !corpus.every((doc) => typeof doc === "string")
    ) {
      throw new Error("corpus must be a list of strings");
    }

    let nDocs = 0;
    let sumDocLen = 0;
    const docFreqCounter: { [key: number]: number } = {};

    for (const doc of corpus) {
      const { indices, tf } = this._tf(doc);
      if (indices.length === 0) continue;
      nDocs += 1;
      sumDocLen += tf.reduce((acc, val) => acc + val, 0);

      for (const idx of indices) {
        docFreqCounter[idx] = (docFreqCounter[idx] || 0) + 1;
      }
    }

    this.doc_freq = docFreqCounter;
    this.n_docs = nDocs;
    this.avgdl = sumDocLen / nDocs;
    return this;
  }

  // Encode documents
  encodeDocuments(texts: string | string[]): SparseVector | SparseVector[] {
    if (this.doc_freq === null || this.n_docs === null || this.avgdl === null) {
      throw new Error("BM25 must be fit before encoding documents");
    }

    if (typeof texts === "string") {
      return this._encodeSingleDocument(texts);
    } else if (Array.isArray(texts)) {
      return texts.map((text) => this._encodeSingleDocument(text));
    } else {
      throw new Error("texts must be a string or list of strings");
    }
  }

  // Encode queries
  encodeQueries(texts: string | string[]): SparseVector | SparseVector[] {
    if (this.doc_freq === null || this.n_docs === null || this.avgdl === null) {
      throw new Error("BM25 must be fit before encoding queries");
    }

    if (typeof texts === "string") {
      return this._encodeSingleQuery(texts);
    } else if (Array.isArray(texts)) {
      return texts.map((text) => this._encodeSingleQuery(text));
    } else {
      throw new Error("texts must be a string or list of strings");
    }
  }

  // Encode a single document
  private _encodeSingleDocument(text: string): SparseVector {
    const { indices, tf } = this._tf(text);
    const tfSum = tf.reduce((acc, val) => acc + val, 0);
    if (this.avgdl == null) {
      throw new Error(
        "avgdl is null this is possibly due to not calling the fiit function first"
      );
    }
    const tfNormed = tf.map(
      (t) => t / (this.k1 * (1 - this.b + this.b * (tfSum / this.avgdl!)) + t)
    );

    return { indices, values: tfNormed };
  }

  // Encode a single query
  private _encodeSingleQuery(text: string): SparseVector {
    const { indices, tf } = this._tf(text);
    const df = indices.map((idx) => this.doc_freq?.[idx] ?? 1);
    if (this.n_docs == null) {
      throw new Error(
        "The number of documents is null, this is possibly due to not calling the fit function first"
      );
    }
    const idf = df.map((d) => Math.log((this.n_docs! + 1) / (d + 0.5)));
    const idfNorm = idf.map((v) => v / idf.reduce((acc, val) => acc + val, 0));
    return { indices, values: idfNorm };
  }

  // Dump parameters to a JSON file
  async dump(path: string) {
    if (this.doc_freq === null || this.n_docs === null || this.avgdl === null) {
      throw new Error("BM25 must be fit before storing params");
    }

    const params = this.getParams();
    await fs.writeFile(path, JSON.stringify(params));
  }

  // Load parameters from a JSON file
  async load(path: string) {
    const params = JSON.parse(await fs.readFile(path, "utf-8"));
    return this.setParams(params);
  }

  // Get BM25 parameters
  getParams(): {
    avgdl: number;
    n_docs: number;
    doc_freq: { indices: number[]; values: number[] };
    b: number;
    k1: number;
    lower_case: boolean;
    remove_punctuation: boolean;
    remove_stopwords: boolean;
    stem: boolean;
    language: string;
  } {
    if (this.doc_freq === null || this.n_docs === null || this.avgdl === null) {
      throw new Error("BM25 must be fit before storing params");
    }
    
    const docFreqPairs = Object.entries(this.doc_freq).map(([idx, val]) => ({
      idx: parseInt(idx),
      val,
    }));
    
    return {
      avgdl: this.avgdl,
      n_docs: this.n_docs,
      doc_freq: {
        indices: docFreqPairs.map((p) => p.idx),
        values: docFreqPairs.map((p) => p.val),
      },
      b: this.b,
      k1: this.k1,
      lower_case: this._tokenizer.lowerCase,
      remove_punctuation: this._tokenizer.removePunctuation,
      remove_stopwords: this._tokenizer.removeStopwords,
      stem: this._tokenizer.stem,
      language: this._tokenizer.language,
    };
  }

  // Set BM25 parameters
  setParams(params: {
    avgdl: number;
    n_docs: number;
    doc_freq: { indices: number[]; values: number[] };
    b: number;
    k1: number;
    lower_case: boolean;
    remove_punctuation: boolean;
    remove_stopwords: boolean;
    stem: boolean;
    language: string;
  }): this {
    this.avgdl = params.avgdl;
    this.n_docs = params.n_docs;
    this.doc_freq = params.doc_freq.indices.reduce((acc, idx, i) => {
      acc[idx] = params.doc_freq.values[i];
      return acc;
    }, {} as { [key: number]: number });
    this.b = params.b;
    this.k1 = params.k1;

    this._tokenizer = new BM25Tokenizer(
      params.lower_case,
      params.remove_punctuation,
      params.remove_stopwords,
      params.stem,
      params.language
    );
    return this;
  }
  
  static async default(filepath: string = ""): Promise<BM25Encoder> {
    // if filepath, then it makes the data persistent to that path, downloads data to the path if needed
    const bm25 = new BM25Encoder();
    const url =
      "https://storage.googleapis.com/pinecone-datasets-dev/bm25_params/msmarco_bm25_params_v4_0_0.json";
    let tempDir: string;
    let tempPath: string;

    // check if filepath is provided
    if (filepath) {
      if (path.extname(filepath).toLowerCase() !== ".json") {
        throw new Error("File is not a JSON file.");
      }

      const valid = await BM25Encoder.fileExists(path.resolve(filepath));
      if (valid) {
        await bm25.load(filepath);
        return bm25;
      } else {
        const dirExists = await BM25Encoder.fileExists(path.dirname(filepath));
        if (!dirExists) {
          await fs.mkdir(path.resolve(path.dirname(filepath)));
        }
        tempPath = path.resolve(filepath);
      }
    } else {
      // create a temporary directory if no filepath
      tempDir = await fs.mkdtemp("bm25_params_");
      tempPath = path.join(tempDir, "msmarco_bm25_params.json");
    }

    try {
      // Download the file from the URL to the path
      console.log("Downloading default params");
      const response = await fetch(url);
      const data = await response.text();
      await fs.writeFile(tempPath, data);

      // Load the BM25 parameters
      await bm25.load(tempPath);
    } catch (error) {
      console.error("Error during BM25 initialization:", error);
    } finally {
      // Cleanup temporary files if filepath was not provided
      if (!filepath) {
        try {
          await fs.rm(tempDir!, { recursive: true, force: true });
        } catch (cleanupError) {
          console.error("Error during cleanup:", cleanupError);
        }
      }
    }

    return bm25;
  }

  // Hash text using MurmurHash3
  private static _hashText(token: string): number {
    return murmurhash.murmur3(token) >>> 0; // Convert to unsigned 32-bit integer
  }

  // Calculate term frequency
  private _tf(text: string): { indices: number[]; tf: number[] } {
    const tokens = this._tokenizer.tokenize(text);
    const counts:Map<number, number> = new Map()

    for (const token of tokens) {
      const idx = BM25Encoder._hashText(token);
      counts.set(idx, (counts.get(idx) || 0) + 1);
    }
    const indices = Array.from(counts.keys())
    const tf = indices.map((idx) => counts.get(idx)!);
    return { indices, tf };
  }
}
