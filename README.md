# Pinecone BM25
The pineconebm25 package is a node package that encodes text for seamless integration with Pinecone's sparse-dense (hybrid) semantic search.

## Installation 
To install use the following command

```bash
npm i pineconebm25
```
## Sparse Encoding 
### BM25
To encode your documents and queries using BM25 as vector for dot product search, you can use the `BM25Encoder` class.

> **_📝 NOTE:_**
> 
>  This implementation of BM25 supports only static document frequency (meaning that the document frequency values are precomputed and fixed, and do not change dynamically based on new documents added to the collection).
>
> When conducting a search, you may come across queries that contain terms not found in the training corpus but are present in the database. To address this scenario, BM25Encoder uses a default document frequency value of 1 when encoding such terms.

```js
import { BM25Encoder } from "pineconebm25";

const corpus = ["The quick brown fox jumps over the lazy dog",
    "The lazy dog is brown",
    "The fox is brown"];

const bm25 = new BM25Encoder();

bm25.fit(corpus);

//Encode a new document (for upsert to Pinecone index)
const doc_sparse_vector = bm25.encodeDocuments("The brown fox is quick");

console.log("Doc sparse vector: ", doc_sparse_vector);

//Encode a query (for search in Pinecone index)
const query_sparse_vector = bm25.encodeQueries("which fox is brown?");

console.log("\nQuery sparse vector", query_sparse_vector);

(async ()=>{
    // store BM25 params as json
    await bm25.dump("./bm25_params.json")

    // Load BM25 params as json 
    const bm25_obj = await bm25.load("./bm25_params.json")
})();

```
#### Load Default Parameters
If you want to use the default parameters for `BM25Encoder`, you can call the `default` method.
The default parameters were fitted on the [MS MARCO](https://microsoft.github.io/msmarco/)  passage ranking dataset.
```js
import { BM25Encoder } from "pineconebm25";
(async ()=>{
    const bm25 = await BM25Encoder.default(); // include filepath to make default params persistent
    console.log("\nEncoded Doc using default params:", bm25.encodeDocuments("The brown fox is quick"));
})();
```

#### BM25 Parameters
The `BM25Encoder` class offers configurable parameters to customize the encoding:

* `b`: Controls document length normalization (default: 0.75).
* `k1`: Controls term frequency saturation (default: 1.2).
* Tokenization Options: Allows customization of the tokenization process, including options for handling case, punctuation, stopwords, stemming, and language selection.

These parameters can be specified when initializing the BM25Encoder class. 