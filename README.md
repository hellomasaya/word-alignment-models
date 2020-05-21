## NLA Assignment 1 - Word Alignment
### Harshita Sharma - 20171099
-------------------------------------
## IBM Model 1

**Data preprocessing:**   
- The datasets are already tokenised. 
- All words are changed to lower case.

**Training the model:**  
- Used `defaultdict` as translation probability table to improve training time and space, where each entry takes key-value pairs in the following format: tef([hindi_word, english_word]) = translation_probability. Using this only relevant pairs of words are looked at.
- For each hindi word in each hindi sentence the corresponding english translated sentence's words are made pairs with.
- The EM algorithm is run i.e. the model is trained for `16 epochs`.
