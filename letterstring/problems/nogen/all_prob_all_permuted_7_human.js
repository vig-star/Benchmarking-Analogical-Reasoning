var all_problems = {
  "np_1": {
    "alph_0": {
      "shuffled_letters": null,
      "shuffled_alphabet": [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z"
      ],
      "succ": [
        {
          "prompt": "[e f g h] &nbsp [e f g i]<br>[f g h i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[j k l m] &nbsp [j k l n]<br>[p q r s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[p q r s] &nbsp [p q r t]<br>[q r s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[g h i j] &nbsp [g h i k]<br>[f g h i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[v w x y] &nbsp [v w x z]<br>[m n o p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[b c d e] &nbsp [b c d f]<br>[q r s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[l m n o] &nbsp [l m n p]<br>[f g h i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[h i j k] &nbsp [h i j l]<br>[l m n o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[v w x y] &nbsp [v w x z]<br>[n o p q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[q r s t] &nbsp [q r s u]<br>[t u v w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "pred": [
        {
          "prompt": "[i j k l] &nbsp [h j k l]<br>[p q r s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[j k l m] &nbsp [i k l m]<br>[d e f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[l m n o] &nbsp [k m n o]<br>[d e f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[f g h i] &nbsp [e g h i]<br>[h i j k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[l m n o] &nbsp [k m n o]<br>[k l m n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[r s t u] &nbsp [q s t u]<br>[l m n o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[m n o p] &nbsp [l n o p]<br>[c d e f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[m n o p] &nbsp [l n o p]<br>[t u v w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[k l m n] &nbsp [j l m n]<br>[o p q r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[l m n o] &nbsp [k m n o]<br>[c d e f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "add_letter": [
        {
          "prompt": "[d e f g] &nbsp [d e f g h]<br>[j k l m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[o p q r] &nbsp [o p q r s]<br>[j k l m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[v w x y] &nbsp [v w x y z]<br>[k l m n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[v w x y] &nbsp [v w x y z]<br>[u v w x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[g h i j] &nbsp [g h i j k]<br>[l m n o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[q r s t] &nbsp [q r s t u]<br>[t u v w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[u v w x] &nbsp [u v w x y]<br>[s t u v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[f g h i] &nbsp [f g h i j]<br>[b c d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[h i j k] &nbsp [h i j k l]<br>[e f g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[m n o p] &nbsp [m n o p q]<br>[p q r s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "remove_redundant": [
        {
          "prompt": "[g g h i j k] &nbsp [g h i j k]<br>[p q q r s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[d e e f g h] &nbsp [d e f g h]<br>[m m n o p q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[n o p q q r] &nbsp [n o p q r]<br>[d e f f g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[t u u v w x] &nbsp [t u v w x]<br>[s t u v w w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[m n o p p q] &nbsp [m n o p q]<br>[r s t t u v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[k l m m n o] &nbsp [k l m n o]<br>[g g h i j k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[h i i j k l] &nbsp [h i j k l]<br>[p q r s t t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[e e f g h i] &nbsp [e f g h i]<br>[n n o p q r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[m n o p p q] &nbsp [m n o p q]<br>[q r s t t u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[v v w x y z] &nbsp [v w x y z]<br>[p q r s t t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "fix_alphabet": [
        {
          "prompt": "[o p q r y] &nbsp [o p q r s]<br>[b c d e a] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[z k l m n] &nbsp [j k l m n]<br>[c d e t g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[o y q r s] &nbsp [o p q r s]<br>[p q r i t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[g r i j k] &nbsp [g h i j k]<br>[e f g h d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[l b n o p] &nbsp [l m n o p]<br>[o p m r s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[d e f n h] &nbsp [d e f g h]<br>[e f j h i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[c d j f g] &nbsp [c d e f g]<br>[q a s t u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[f g h i r] &nbsp [f g h i j]<br>[b v d e f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[n o d q r] &nbsp [n o p q r]<br>[g h i j f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[n o b q r] &nbsp [n o p q r]<br>[r i t u v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "sort": [
        {
          "prompt": "[e b c d a] &nbsp [a b c d e]<br>[o m n l p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[r o p q n] &nbsp [n o p q r]<br>[l n m o p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[x v w u y] &nbsp [u v w x y]<br>[i j l k m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[p o n q r] &nbsp [n o p q r]<br>[t w v u x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[q p o r s] &nbsp [o p q r s]<br>[n k l m j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[c d e g f] &nbsp [c d e f g]<br>[u v y x w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[i j m l k] &nbsp [i j k l m]<br>[g f e h i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[c d e g f] &nbsp [c d e f g]<br>[p t r s q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[b c f e d] &nbsp [b c d e f]<br>[v t u s w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[h j i k l] &nbsp [h i j k l]<br>[l k m n o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "attn": [
        {
          "prompt": "[h i j k] &nbsp [h i j l]<br>[l m n o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[f g h i] &nbsp [f g h j]<br>[u v w x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[e f g h] &nbsp [e f g i]<br>[q r s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[d e f g] &nbsp [d e f h]<br>[k l m n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[r s t u] &nbsp [r s t v]<br>[a b c d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[q r s t] &nbsp [q r s u]<br>[b c d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[q r s t] &nbsp [q r s u]<br>[i j k l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[b c d e] &nbsp [b c d f]<br>[q r s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[d e f g] &nbsp [d e f h]<br>[o p q r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[g h i j] &nbsp [g h i k]<br>[t u v w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ]
    },
    "alph_1": {
      "shuffled_letters": null,
      "shuffled_alphabet": [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z"
      ],
      "succ": [
        {
          "prompt": "[d e f g] &nbsp [d e f h]<br>[b c d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[l m n o] &nbsp [l m n p]<br>[g h i j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[j k l m] &nbsp [j k l n]<br>[i j k l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[d e f g] &nbsp [d e f h]<br>[v w x y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[a b c d] &nbsp [a b c e]<br>[p q r s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[m n o p] &nbsp [m n o q]<br>[i j k l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[j k l m] &nbsp [j k l n]<br>[o p q r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[j k l m] &nbsp [j k l n]<br>[q r s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[f g h i] &nbsp [f g h j]<br>[a b c d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[m n o p] &nbsp [m n o q]<br>[l m n o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "pred": [
        {
          "prompt": "[j k l m] &nbsp [i k l m]<br>[i j k l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[n o p q] &nbsp [m o p q]<br>[u v w x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[h i j k] &nbsp [g i j k]<br>[d e f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[g h i j] &nbsp [f h i j]<br>[w x y z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[j k l m] &nbsp [i k l m]<br>[m n o p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[e f g h] &nbsp [d f g h]<br>[u v w x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[r s t u] &nbsp [q s t u]<br>[c d e f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[f g h i] &nbsp [e g h i]<br>[q r s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[g h i j] &nbsp [f h i j]<br>[o p q r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[p q r s] &nbsp [o q r s]<br>[r s t u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "add_letter": [
        {
          "prompt": "[q r s t] &nbsp [q r s t u]<br>[g h i j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[k l m n] &nbsp [k l m n o]<br>[i j k l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[a b c d] &nbsp [a b c d e]<br>[n o p q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[i j k l] &nbsp [i j k l m]<br>[e f g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[v w x y] &nbsp [v w x y z]<br>[k l m n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[j k l m] &nbsp [j k l m n]<br>[m n o p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[e f g h] &nbsp [e f g h i]<br>[t u v w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[q r s t] &nbsp [q r s t u]<br>[u v w x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[h i j k] &nbsp [h i j k l]<br>[a b c d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[u v w x] &nbsp [u v w x y]<br>[t u v w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "remove_redundant": [
        {
          "prompt": "[s t u v v w] &nbsp [s t u v w]<br>[d e f f g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[r s t u v v] &nbsp [r s t u v]<br>[n o p q r r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[i j k k l m] &nbsp [i j k l m]<br>[k l m m n o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[a b c c d e] &nbsp [a b c d e]<br>[c d e f g g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[p p q r s t] &nbsp [p q r s t]<br>[e f g h h i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[a b b c d e] &nbsp [a b c d e]<br>[o p q r r s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[k k l m n o] &nbsp [k l m n o]<br>[r s s t u v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[i j k k l m] &nbsp [i j k l m]<br>[f g h i j j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[j j k l m n] &nbsp [j k l m n]<br>[e f g g h i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[d e f f g h] &nbsp [d e f g h]<br>[q r s t u u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "fix_alphabet": [
        {
          "prompt": "[v w x d z] &nbsp [v w x y z]<br>[i j q l m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[z o p q r] &nbsp [n o p q r]<br>[o d e f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[a b c d u] &nbsp [a b c d e]<br>[s t u v j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[j u l m n] &nbsp [j k l m n]<br>[s t u v m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[m n o e q] &nbsp [m n o p q]<br>[b c d e l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[o p q j s] &nbsp [o p q r s]<br>[z i j k l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[o p q z s] &nbsp [o p q r s]<br>[w n o p q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[r i j k l] &nbsp [h i j k l]<br>[b c d e i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[i j h l m] &nbsp [i j k l m]<br>[o p q r j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[v w l y z] &nbsp [v w x y z]<br>[s t k v w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "sort": [
        {
          "prompt": "[c b a d e] &nbsp [a b c d e]<br>[e h g f i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[e f i h g] &nbsp [e f g h i]<br>[p m n o l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[l k m n o] &nbsp [k l m n o]<br>[f h g i j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[b a c d e] &nbsp [a b c d e]<br>[d g f e h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[g f h i j] &nbsp [f g h i j]<br>[w v x y z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[p n o m q] &nbsp [m n o p q]<br>[q u s t r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[p t r s q] &nbsp [p q r s t]<br>[f g h j i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[q o p n r] &nbsp [n o p q r]<br>[g d e f c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[t u v x w] &nbsp [t u v w x]<br>[j m l k n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[a d c b e] &nbsp [a b c d e]<br>[k o m n l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "attn": [
        {
          "prompt": "[k l m n] &nbsp [k l m o]<br>[a b c d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[a b c d] &nbsp [a b c e]<br>[u v w x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[u v w x] &nbsp [u v w y]<br>[h i j k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[m n o p] &nbsp [m n o q]<br>[k l m n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[r s t u] &nbsp [r s t v]<br>[n o p q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[d e f g] &nbsp [d e f h]<br>[v w x y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[q r s t] &nbsp [q r s u]<br>[f g h i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[p q r s] &nbsp [p q r t]<br>[n o p q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[i j k l] &nbsp [i j k m]<br>[a b c d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[a b c d] &nbsp [a b c e]<br>[g h i j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ]
    },
    "alph_2": {
      "shuffled_letters": null,
      "shuffled_alphabet": [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z"
      ],
      "succ": [
        {
          "prompt": "[s t u v] &nbsp [s t u w]<br>[u v w x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[l m n o] &nbsp [l m n p]<br>[p q r s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[r s t u] &nbsp [r s t v]<br>[o p q r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[l m n o] &nbsp [l m n p]<br>[j k l m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[f g h i] &nbsp [f g h j]<br>[g h i j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[f g h i] &nbsp [f g h j]<br>[e f g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[e f g h] &nbsp [e f g i]<br>[q r s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[d e f g] &nbsp [d e f h]<br>[q r s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[n o p q] &nbsp [n o p r]<br>[g h i j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[l m n o] &nbsp [l m n p]<br>[i j k l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "pred": [
        {
          "prompt": "[t u v w] &nbsp [s u v w]<br>[h i j k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[l m n o] &nbsp [k m n o]<br>[e f g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[r s t u] &nbsp [q s t u]<br>[i j k l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[g h i j] &nbsp [f h i j]<br>[s t u v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[t u v w] &nbsp [s u v w]<br>[u v w x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[p q r s] &nbsp [o q r s]<br>[k l m n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[v w x y] &nbsp [u w x y]<br>[e f g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[s t u v] &nbsp [r t u v]<br>[i j k l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[t u v w] &nbsp [s u v w]<br>[v w x y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[p q r s] &nbsp [o q r s]<br>[i j k l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "add_letter": [
        {
          "prompt": "[k l m n] &nbsp [k l m n o]<br>[n o p q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[u v w x] &nbsp [u v w x y]<br>[l m n o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[h i j k] &nbsp [h i j k l]<br>[v w x y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[v w x y] &nbsp [v w x y z]<br>[n o p q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[o p q r] &nbsp [o p q r s]<br>[f g h i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[m n o p] &nbsp [m n o p q]<br>[c d e f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[m n o p] &nbsp [m n o p q]<br>[g h i j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[t u v w] &nbsp [t u v w x]<br>[b c d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[p q r s] &nbsp [p q r s t]<br>[t u v w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[v w x y] &nbsp [v w x y z]<br>[u v w x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "remove_redundant": [
        {
          "prompt": "[p q r s t t] &nbsp [p q r s t]<br>[e e f g h i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[b c d e f f] &nbsp [b c d e f]<br>[t u v w w x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[l l m n o p] &nbsp [l m n o p]<br>[u v w x x y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[c d e f f g] &nbsp [c d e f g]<br>[l l m n o p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[n o p q q r] &nbsp [n o p q r]<br>[l m n o o p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[u v v w x y] &nbsp [u v w x y]<br>[h h i j k l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[d e f g g h] &nbsp [d e f g h]<br>[a a b c d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[q r s t t u] &nbsp [q r s t u]<br>[m n o p p q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[s t u u v w] &nbsp [s t u v w]<br>[g h i j k k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[u v w x y y] &nbsp [u v w x y]<br>[m n o o p q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "fix_alphabet": [
        {
          "prompt": "[b j d e f] &nbsp [b c d e f]<br>[e f w h i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[u v w x h] &nbsp [u v w x y]<br>[h i y k l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[c h i j k] &nbsp [g h i j k]<br>[r g h i j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[l i n o p] &nbsp [l m n o p]<br>[o d e f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[w j k l m] &nbsp [i j k l m]<br>[f g h e j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[a b c d o] &nbsp [a b c d e]<br>[f g w i j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[f g h i l] &nbsp [f g h i j]<br>[p q r w t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[k l m n y] &nbsp [k l m n o]<br>[h i j y l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[d r s t u] &nbsp [q r s t u]<br>[u v w q y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[o p a r s] &nbsp [o p q r s]<br>[v w j y z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "sort": [
        {
          "prompt": "[l j k i m] &nbsp [i j k l m]<br>[g i h j k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[m k l j n] &nbsp [j k l m n]<br>[u v w y x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[b d c e f] &nbsp [b c d e f]<br>[k i j h l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[q n o p m] &nbsp [m n o p q]<br>[n o p r q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[m p o n q] &nbsp [m n o p q]<br>[b c d f e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[n p o q r] &nbsp [n o p q r]<br>[q r t s u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[s r q t u] &nbsp [q r s t u]<br>[t v u w x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[m o n p q] &nbsp [m n o p q]<br>[a d c b e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[r q s t u] &nbsp [q r s t u]<br>[a b c e d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[s r t u v] &nbsp [r s t u v]<br>[d g f e h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "attn": [
        {
          "prompt": "[a b c d] &nbsp [a b c e]<br>[s t u v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[e f g h] &nbsp [e f g i]<br>[u v w x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[s t u v] &nbsp [s t u w]<br>[d e f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[c d e f] &nbsp [c d e g]<br>[b c d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[v w x y] &nbsp [v w x z]<br>[h i j k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[j k l m] &nbsp [j k l n]<br>[c d e f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[d e f g] &nbsp [d e f h]<br>[p q r s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[j k l m] &nbsp [j k l n]<br>[p q r s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[o p q r] &nbsp [o p q s]<br>[f g h i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[u v w x] &nbsp [u v w y]<br>[i j k l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ]
    },
    "alph_3": {
      "shuffled_letters": null,
      "shuffled_alphabet": [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z"
      ],
      "succ": [
        {
          "prompt": "[m n o p] &nbsp [m n o q]<br>[f g h i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[q r s t] &nbsp [q r s u]<br>[u v w x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[a b c d] &nbsp [a b c e]<br>[f g h i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[j k l m] &nbsp [j k l n]<br>[u v w x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[a b c d] &nbsp [a b c e]<br>[c d e f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[k l m n] &nbsp [k l m o]<br>[s t u v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[g h i j] &nbsp [g h i k]<br>[n o p q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[q r s t] &nbsp [q r s u]<br>[l m n o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[h i j k] &nbsp [h i j l]<br>[c d e f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[q r s t] &nbsp [q r s u]<br>[e f g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "pred": [
        {
          "prompt": "[j k l m] &nbsp [i k l m]<br>[n o p q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[g h i j] &nbsp [f h i j]<br>[t u v w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[j k l m] &nbsp [i k l m]<br>[k l m n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[t u v w] &nbsp [s u v w]<br>[l m n o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[i j k l] &nbsp [h j k l]<br>[v w x y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[g h i j] &nbsp [f h i j]<br>[n o p q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[u v w x] &nbsp [t v w x]<br>[q r s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[l m n o] &nbsp [k m n o]<br>[i j k l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[i j k l] &nbsp [h j k l]<br>[g h i j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[d e f g] &nbsp [c e f g]<br>[p q r s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "add_letter": [
        {
          "prompt": "[a b c d] &nbsp [a b c d e]<br>[t u v w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[p q r s] &nbsp [p q r s t]<br>[m n o p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[h i j k] &nbsp [h i j k l]<br>[u v w x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[m n o p] &nbsp [m n o p q]<br>[s t u v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[i j k l] &nbsp [i j k l m]<br>[j k l m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[r s t u] &nbsp [r s t u v]<br>[e f g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[q r s t] &nbsp [q r s t u]<br>[s t u v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[n o p q] &nbsp [n o p q r]<br>[t u v w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[a b c d] &nbsp [a b c d e]<br>[s t u v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[j k l m] &nbsp [j k l m n]<br>[e f g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "remove_redundant": [
        {
          "prompt": "[e f g h h i] &nbsp [e f g h i]<br>[c d d e f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[l m n o p p] &nbsp [l m n o p]<br>[v w w x y z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[m n n o p q] &nbsp [m n o p q]<br>[o p q r s s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[j k k l m n] &nbsp [j k l m n]<br>[n o o p q r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[f g h h i j] &nbsp [f g h i j]<br>[m n o o p q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[h i j j k l] &nbsp [h i j k l]<br>[f f g h i j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[i j j k l m] &nbsp [i j k l m]<br>[t t u v w x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[b c d e f f] &nbsp [b c d e f]<br>[t u v v w x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[r r s t u v] &nbsp [r s t u v]<br>[v v w x y z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[s t u v v w] &nbsp [s t u v w]<br>[e f f g h i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "fix_alphabet": [
        {
          "prompt": "[g h p j k] &nbsp [g h i j k]<br>[c d s f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[q v w x y] &nbsp [u v w x y]<br>[o l q r s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[r s t b v] &nbsp [r s t u v]<br>[d v f g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[m j o p q] &nbsp [m n o p q]<br>[q j s t u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[n o p u r] &nbsp [n o p q r]<br>[q r f t u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[s t j v w] &nbsp [s t u v w]<br>[t u v w s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[a u c d e] &nbsp [a b c d e]<br>[f g h t j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[u v w r y] &nbsp [u v w x y]<br>[v m n o p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[h i d k l] &nbsp [h i j k l]<br>[d e j g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[n o p q l] &nbsp [n o p q r]<br>[z g h i j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "sort": [
        {
          "prompt": "[o l m n k] &nbsp [k l m n o]<br>[d e h g f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[d g f e h] &nbsp [d e f g h]<br>[v s t u r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[v s t u r] &nbsp [r s t u v]<br>[u v x w y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[t s r u v] &nbsp [r s t u v]<br>[c d e g f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[v y x w z] &nbsp [v w x y z]<br>[k j l m n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[h g i j k] &nbsp [g h i j k]<br>[u x w v y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[j i h k l] &nbsp [h i j k l]<br>[f e g h i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[b e d c f] &nbsp [b c d e f]<br>[j k l n m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[e f h g i] &nbsp [e f g h i]<br>[n m o p q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[j h i g k] &nbsp [g h i j k]<br>[j k l n m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "attn": [
        {
          "prompt": "[i j k l] &nbsp [i j k m]<br>[f g h i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[e f g h] &nbsp [e f g i]<br>[r s t u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[m n o p] &nbsp [m n o q]<br>[h i j k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[q r s t] &nbsp [q r s u]<br>[f g h i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[r s t u] &nbsp [r s t v]<br>[h i j k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[v w x y] &nbsp [v w x z]<br>[q r s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[i j k l] &nbsp [i j k m]<br>[l m n o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[u v w x] &nbsp [u v w y]<br>[g h i j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[l m n o] &nbsp [l m n p]<br>[t u v w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[i j k l] &nbsp [i j k m]<br>[r s t u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ]
    },
    "alph_4": {
      "shuffled_letters": null,
      "shuffled_alphabet": [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z"
      ],
      "succ": [
        {
          "prompt": "[j k l m] &nbsp [j k l n]<br>[l m n o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[u v w x] &nbsp [u v w y]<br>[g h i j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[t u v w] &nbsp [t u v x]<br>[a b c d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[h i j k] &nbsp [h i j l]<br>[i j k l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[h i j k] &nbsp [h i j l]<br>[d e f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[j k l m] &nbsp [j k l n]<br>[e f g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[b c d e] &nbsp [b c d f]<br>[q r s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[l m n o] &nbsp [l m n p]<br>[o p q r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[c d e f] &nbsp [c d e g]<br>[o p q r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[s t u v] &nbsp [s t u w]<br>[c d e f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "pred": [
        {
          "prompt": "[u v w x] &nbsp [t v w x]<br>[j k l m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[l m n o] &nbsp [k m n o]<br>[c d e f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[j k l m] &nbsp [i k l m]<br>[u v w x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[v w x y] &nbsp [u w x y]<br>[g h i j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[h i j k] &nbsp [g i j k]<br>[i j k l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[r s t u] &nbsp [q s t u]<br>[s t u v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[j k l m] &nbsp [i k l m]<br>[o p q r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[r s t u] &nbsp [q s t u]<br>[v w x y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[v w x y] &nbsp [u w x y]<br>[f g h i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[p q r s] &nbsp [o q r s]<br>[u v w x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "add_letter": [
        {
          "prompt": "[o p q r] &nbsp [o p q r s]<br>[j k l m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[o p q r] &nbsp [o p q r s]<br>[n o p q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[r s t u] &nbsp [r s t u v]<br>[m n o p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[f g h i] &nbsp [f g h i j]<br>[m n o p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[e f g h] &nbsp [e f g h i]<br>[t u v w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[v w x y] &nbsp [v w x y z]<br>[h i j k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[q r s t] &nbsp [q r s t u]<br>[f g h i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[t u v w] &nbsp [t u v w x]<br>[p q r s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[l m n o] &nbsp [l m n o p]<br>[i j k l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[d e f g] &nbsp [d e f g h]<br>[s t u v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "remove_redundant": [
        {
          "prompt": "[l m n o p p] &nbsp [l m n o p]<br>[t u v w w x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[n n o p q r] &nbsp [n o p q r]<br>[t t u v w x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[u v w x y y] &nbsp [u v w x y]<br>[c d e e f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[q r s t u u] &nbsp [q r s t u]<br>[b c d d e f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[p p q r s t] &nbsp [p q r s t]<br>[r r s t u v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[j j k l m n] &nbsp [j k l m n]<br>[t u v w x x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[t u v w w x] &nbsp [t u v w x]<br>[f g h i i j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[m n o p q q] &nbsp [m n o p q]<br>[h i i j k l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[r s t t u v] &nbsp [r s t u v]<br>[a b c d e e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[n n o p q r] &nbsp [n o p q r]<br>[e e f g h i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "fix_alphabet": [
        {
          "prompt": "[o k l m n] &nbsp [j k l m n]<br>[u v w x b] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[h i x k l] &nbsp [h i j k l]<br>[m n o p v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[b f g h i] &nbsp [e f g h i]<br>[h i r k l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[r s c u v] &nbsp [r s t u v]<br>[v d x y z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[o p q r v] &nbsp [o p q r s]<br>[l m w o p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[b j k l m] &nbsp [i j k l m]<br>[p e r s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[g r i j k] &nbsp [g h i j k]<br>[f o p q r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[t b c d e] &nbsp [a b c d e]<br>[g m i j k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[o p i r s] &nbsp [o p q r s]<br>[a k c d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[a b s d e] &nbsp [a b c d e]<br>[c h i j k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "sort": [
        {
          "prompt": "[w u v t x] &nbsp [t u v w x]<br>[r s u t v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[h i l k j] &nbsp [h i j k l]<br>[k h i j g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[j l k m n] &nbsp [j k l m n]<br>[a c b d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[d e f h g] &nbsp [d e f g h]<br>[r v t u s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[j k n m l] &nbsp [j k l m n]<br>[v w z y x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[m p o n q] &nbsp [m n o p q]<br>[h e f g d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[g i h j k] &nbsp [g h i j k]<br>[q n o p m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[j k m l n] &nbsp [j k l m n]<br>[t r s q u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[o m n l p] &nbsp [l m n o p]<br>[h i l k j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[h g f i j] &nbsp [f g h i j]<br>[d f e g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "attn": [
        {
          "prompt": "[u v w x] &nbsp [u v w y]<br>[t u v w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[d e f g] &nbsp [d e f h]<br>[i j k l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[d e f g] &nbsp [d e f h]<br>[u v w x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[e f g h] &nbsp [e f g i]<br>[c d e f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[g h i j] &nbsp [g h i k]<br>[o p q r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[d e f g] &nbsp [d e f h]<br>[b c d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[t u v w] &nbsp [t u v x]<br>[n o p q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[i j k l] &nbsp [i j k m]<br>[j k l m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[m n o p] &nbsp [m n o q]<br>[n o p q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[h i j k] &nbsp [h i j l]<br>[s t u v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ]
    },
    "alph_5": {
      "shuffled_letters": null,
      "shuffled_alphabet": [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z"
      ],
      "succ": [
        {
          "prompt": "[c d e f] &nbsp [c d e g]<br>[o p q r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[b c d e] &nbsp [b c d f]<br>[n o p q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[p q r s] &nbsp [p q r t]<br>[q r s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[d e f g] &nbsp [d e f h]<br>[m n o p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[s t u v] &nbsp [s t u w]<br>[h i j k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[s t u v] &nbsp [s t u w]<br>[d e f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[f g h i] &nbsp [f g h j]<br>[c d e f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[n o p q] &nbsp [n o p r]<br>[g h i j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[j k l m] &nbsp [j k l n]<br>[b c d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[b c d e] &nbsp [b c d f]<br>[a b c d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "pred": [
        {
          "prompt": "[w x y z] &nbsp [v x y z]<br>[j k l m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[u v w x] &nbsp [t v w x]<br>[q r s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[v w x y] &nbsp [u w x y]<br>[g h i j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[u v w x] &nbsp [t v w x]<br>[m n o p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[v w x y] &nbsp [u w x y]<br>[p q r s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[o p q r] &nbsp [n p q r]<br>[k l m n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[o p q r] &nbsp [n p q r]<br>[n o p q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[h i j k] &nbsp [g i j k]<br>[m n o p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[f g h i] &nbsp [e g h i]<br>[h i j k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[v w x y] &nbsp [u w x y]<br>[o p q r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "add_letter": [
        {
          "prompt": "[v w x y] &nbsp [v w x y z]<br>[j k l m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[g h i j] &nbsp [g h i j k]<br>[p q r s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[k l m n] &nbsp [k l m n o]<br>[r s t u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[n o p q] &nbsp [n o p q r]<br>[t u v w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[g h i j] &nbsp [g h i j k]<br>[u v w x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[c d e f] &nbsp [c d e f g]<br>[r s t u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[f g h i] &nbsp [f g h i j]<br>[c d e f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[g h i j] &nbsp [g h i j k]<br>[l m n o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[r s t u] &nbsp [r s t u v]<br>[e f g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[d e f g] &nbsp [d e f g h]<br>[o p q r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "remove_redundant": [
        {
          "prompt": "[g g h i j k] &nbsp [g h i j k]<br>[k l m n n o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[s t t u v w] &nbsp [s t u v w]<br>[u v w x x y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[t u u v w x] &nbsp [t u v w x]<br>[e f g h i i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[q r s t t u] &nbsp [q r s t u]<br>[n o o p q r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[n o p p q r] &nbsp [n o p q r]<br>[c d e e f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[c d e f g g] &nbsp [c d e f g]<br>[e f g h i i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[r r s t u v] &nbsp [r s t u v]<br>[j k l l m n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[p q r s s t] &nbsp [p q r s t]<br>[c d e f f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[h i i j k l] &nbsp [h i j k l]<br>[l m n o p p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[t u v w x x] &nbsp [t u v w x]<br>[g h i j k k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "fix_alphabet": [
        {
          "prompt": "[k l y n o] &nbsp [k l m n o]<br>[c g h i j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[i n o p q] &nbsp [m n o p q]<br>[j k l m g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[m n o p w] &nbsp [m n o p q]<br>[q r g t u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[d r f g h] &nbsp [d e f g h]<br>[j k t m n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[c d e q g] &nbsp [c d e f g]<br>[b c m e f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[s t u v j] &nbsp [s t u v w]<br>[b c i e f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[e g h i j] &nbsp [f g h i j]<br>[e f g j i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[o c q r s] &nbsp [o p q r s]<br>[a b c d v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[p q r s d] &nbsp [p q r s t]<br>[u v w e y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[d e a g h] &nbsp [d e f g h]<br>[l m n j p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "sort": [
        {
          "prompt": "[j g h i f] &nbsp [f g h i j]<br>[l k m n o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[l j k i m] &nbsp [i j k l m]<br>[q p r s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[g h i k j] &nbsp [g h i j k]<br>[c b a d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[m j k l i] &nbsp [i j k l m]<br>[q s r t u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[v y x w z] &nbsp [v w x y z]<br>[u v w y x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[m k l j n] &nbsp [j k l m n]<br>[p o q r s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[q p r s t] &nbsp [p q r s t]<br>[f e d g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[r v t u s] &nbsp [r s t u v]<br>[k n m l o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[u x w v y] &nbsp [u v w x y]<br>[t r s q u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[t v u w x] &nbsp [t u v w x]<br>[k m l n o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "attn": [
        {
          "prompt": "[r s t u] &nbsp [r s t v]<br>[j k l m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[h i j k] &nbsp [h i j l]<br>[d e f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[u v w x] &nbsp [u v w y]<br>[i j k l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[p q r s] &nbsp [p q r t]<br>[b c d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[o p q r] &nbsp [o p q s]<br>[h i j k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[r s t u] &nbsp [r s t v]<br>[p q r s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[o p q r] &nbsp [o p q s]<br>[e f g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[k l m n] &nbsp [k l m o]<br>[i j k l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[j k l m] &nbsp [j k l n]<br>[u v w x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[c d e f] &nbsp [c d e g]<br>[f g h i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ]
    },
    "alph_6": {
      "shuffled_letters": null,
      "shuffled_alphabet": [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z"
      ],
      "succ": [
        {
          "prompt": "[b c d e] &nbsp [b c d f]<br>[i j k l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[o p q r] &nbsp [o p q s]<br>[e f g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[u v w x] &nbsp [u v w y]<br>[h i j k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[e f g h] &nbsp [e f g i]<br>[p q r s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[b c d e] &nbsp [b c d f]<br>[s t u v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[u v w x] &nbsp [u v w y]<br>[o p q r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[q r s t] &nbsp [q r s u]<br>[u v w x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[n o p q] &nbsp [n o p r]<br>[h i j k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[i j k l] &nbsp [i j k m]<br>[j k l m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[f g h i] &nbsp [f g h j]<br>[s t u v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "pred": [
        {
          "prompt": "[s t u v] &nbsp [r t u v]<br>[d e f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[n o p q] &nbsp [m o p q]<br>[u v w x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[g h i j] &nbsp [f h i j]<br>[e f g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[c d e f] &nbsp [b d e f]<br>[l m n o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[p q r s] &nbsp [o q r s]<br>[h i j k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[o p q r] &nbsp [n p q r]<br>[b c d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[t u v w] &nbsp [s u v w]<br>[f g h i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[q r s t] &nbsp [p r s t]<br>[m n o p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[s t u v] &nbsp [r t u v]<br>[q r s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[g h i j] &nbsp [f h i j]<br>[w x y z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "add_letter": [
        {
          "prompt": "[i j k l] &nbsp [i j k l m]<br>[j k l m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[d e f g] &nbsp [d e f g h]<br>[f g h i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[j k l m] &nbsp [j k l m n]<br>[u v w x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[c d e f] &nbsp [c d e f g]<br>[k l m n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[e f g h] &nbsp [e f g h i]<br>[d e f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[t u v w] &nbsp [t u v w x]<br>[n o p q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[r s t u] &nbsp [r s t u v]<br>[m n o p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[h i j k] &nbsp [h i j k l]<br>[t u v w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[q r s t] &nbsp [q r s t u]<br>[m n o p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[j k l m] &nbsp [j k l m n]<br>[e f g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "remove_redundant": [
        {
          "prompt": "[q q r s t u] &nbsp [q r s t u]<br>[d e f g g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[t u v w w x] &nbsp [t u v w x]<br>[k l m m n o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[g h i j k k] &nbsp [g h i j k]<br>[a b b c d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[d e f g g h] &nbsp [d e f g h]<br>[a b b c d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[i i j k l m] &nbsp [i j k l m]<br>[q r s s t u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[p q q r s t] &nbsp [p q r s t]<br>[o p p q r s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[g h h i j k] &nbsp [g h i j k]<br>[l m n o p p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[s t u v v w] &nbsp [s t u v w]<br>[c d e f g g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[d e e f g h] &nbsp [d e f g h]<br>[g h h i j k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[e e f g h i] &nbsp [e f g h i]<br>[i j k l m m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "fix_alphabet": [
        {
          "prompt": "[o t u v w] &nbsp [s t u v w]<br>[q r s t j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[t u v w e] &nbsp [t u v w x]<br>[q r s t o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[g h i j a] &nbsp [g h i j k]<br>[f g h i y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[i j k l e] &nbsp [i j k l m]<br>[s b c d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[o a q r s] &nbsp [o p q r s]<br>[v w x y j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[d s f g h] &nbsp [d e f g h]<br>[s t u b w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[q t u v w] &nbsp [s t u v w]<br>[v w x m z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[g h i n k] &nbsp [g h i j k]<br>[k l u n o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[e f g h l] &nbsp [e f g h i]<br>[c d l f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[o p q r c] &nbsp [o p q r s]<br>[a v c d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "sort": [
        {
          "prompt": "[r u t s v] &nbsp [r s t u v]<br>[u y w x v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[m l k n o] &nbsp [k l m n o]<br>[q n o p m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[u r s t q] &nbsp [q r s t u]<br>[p o n q r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[t w v u x] &nbsp [t u v w x]<br>[r v t u s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[i h g j k] &nbsp [g h i j k]<br>[m p o n q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[u v y x w] &nbsp [u v w x y]<br>[h l j k i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[r p q o s] &nbsp [o p q r s]<br>[q t s r u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[h k j i l] &nbsp [h i j k l]<br>[k h i j g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[r s u t v] &nbsp [r s t u v]<br>[s p q r o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[p m n o l] &nbsp [l m n o p]<br>[l k j m n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "attn": [
        {
          "prompt": "[p q r s] &nbsp [p q r t]<br>[i j k l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[b c d e] &nbsp [b c d f]<br>[p q r s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[h i j k] &nbsp [h i j l]<br>[m n o p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[q r s t] &nbsp [q r s u]<br>[t u v w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[l m n o] &nbsp [l m n p]<br>[g h i j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[h i j k] &nbsp [h i j l]<br>[e f g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[r s t u] &nbsp [r s t v]<br>[e f g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[q r s t] &nbsp [q r s u]<br>[p q r s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[q r s t] &nbsp [q r s u]<br>[h i j k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[h i j k] &nbsp [h i j l]<br>[b c d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ]
    }
  },
  "np_2": {
    "alph_0": {
      "shuffled_letters": [
        "u",
        "b"
      ],
      "shuffled_alphabet": [
        "a",
        "u",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "b",
        "v",
        "w",
        "x",
        "y",
        "z"
      ],
      "succ": [
        {
          "prompt": "[f g h i] &nbsp [f g h j]<br>[r s t b] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[r s t b] &nbsp [r s t v]<br>[g h i j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[v w x y] &nbsp [v w x z]<br>[r s t b] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[q r s t] &nbsp [q r s b]<br>[a u c d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[h i j k] &nbsp [h i j l]<br>[a u c d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[t b v w] &nbsp [t b v x]<br>[h i j k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[a u c d] &nbsp [a u c e]<br>[c d e f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[c d e f] &nbsp [c d e g]<br>[b v w x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[k l m n] &nbsp [k l m o]<br>[u c d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[l m n o] &nbsp [l m n p]<br>[q r s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "pred": [
        {
          "prompt": "[s t b v] &nbsp [r t b v]<br>[k l m n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[t b v w] &nbsp [s b v w]<br>[e f g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[p q r s] &nbsp [o q r s]<br>[c d e f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[v w x y] &nbsp [b w x y]<br>[h i j k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[u c d e] &nbsp [a c d e]<br>[b v w x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[u c d e] &nbsp [a c d e]<br>[c d e f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[f g h i] &nbsp [e g h i]<br>[u c d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[u c d e] &nbsp [a c d e]<br>[q r s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[r s t b] &nbsp [q s t b]<br>[o p q r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[n o p q] &nbsp [m o p q]<br>[u c d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "add_letter": [
        {
          "prompt": "[p q r s] &nbsp [p q r s t]<br>[a u c d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[r s t b] &nbsp [r s t b v]<br>[d e f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[r s t b] &nbsp [r s t b v]<br>[e f g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[b v w x] &nbsp [b v w x y]<br>[t b v w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[g h i j] &nbsp [g h i j k]<br>[u c d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[l m n o] &nbsp [l m n o p]<br>[b v w x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[s t b v] &nbsp [s t b v w]<br>[q r s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[t b v w] &nbsp [t b v w x]<br>[q r s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[r s t b] &nbsp [r s t b v]<br>[h i j k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[a u c d] &nbsp [a u c d e]<br>[h i j k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "remove_redundant": [
        {
          "prompt": "[u c d e f f] &nbsp [u c d e f]<br>[q r s t t b] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[f f g h i j] &nbsp [f g h i j]<br>[t b v v w x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[a u c c d e] &nbsp [a u c d e]<br>[n n o p q r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[t t b v w x] &nbsp [t b v w x]<br>[o p q r r s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[v w x y y z] &nbsp [v w x y z]<br>[b v w x x y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[h i j j k l] &nbsp [h i j k l]<br>[a u c c d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[s t b v w w] &nbsp [s t b v w]<br>[r s t t b v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[t t b v w x] &nbsp [t b v w x]<br>[j k l l m n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[t b v w w x] &nbsp [t b v w x]<br>[a u c d e e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[f g g h i j] &nbsp [f g h i j]<br>[a a u c d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "fix_alphabet": [
        {
          "prompt": "[a u c f e] &nbsp [a u c d e]<br>[x d e f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[b v w x m] &nbsp [b v w x y]<br>[r y t b v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[b c d e f] &nbsp [u c d e f]<br>[a u o d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[t b v r x] &nbsp [t b v w x]<br>[m w o p q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[e f j h i] &nbsp [e f g h i]<br>[a f c d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[u c d e y] &nbsp [u c d e f]<br>[a v c d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[a u n d e] &nbsp [a u c d e]<br>[d o f g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[a s t b v] &nbsp [r s t b v]<br>[n o j q r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[p q r h t] &nbsp [p q r s t]<br>[t b v w k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[e f g y i] &nbsp [e f g h i]<br>[f u c d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "sort": [
        {
          "prompt": "[e c d u f] &nbsp [u c d e f]<br>[x b v w t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[f e d g h] &nbsp [d e f g h]<br>[a u c e d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[a c u d e] &nbsp [a u c d e]<br>[p q t s r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[s r q t b] &nbsp [q r s t b]<br>[r s b t v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[f c d e u] &nbsp [u c d e f]<br>[h e f g d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[a d c u e] &nbsp [a u c d e]<br>[w b v t x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[u c f e d] &nbsp [u c d e f]<br>[q t s r b] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[u c f e d] &nbsp [u c d e f]<br>[a u d c e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[r v t b s] &nbsp [r s t b v]<br>[t b v x w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[u c e d f] &nbsp [u c d e f]<br>[d f e g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "attn": [
        {
          "prompt": "[v w x y] &nbsp [v w x z]<br>[a u c d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[r s t b] &nbsp [r s t v]<br>[t b v w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[g h i j] &nbsp [g h i k]<br>[s t b v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[r s t b] &nbsp [r s t v]<br>[g h i j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[a u c d] &nbsp [a u c e]<br>[p q r s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[o p q r] &nbsp [o p q s]<br>[u c d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[m n o p] &nbsp [m n o q]<br>[a u c d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[a u c d] &nbsp [a u c e]<br>[s t b v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[a u c d] &nbsp [a u c e]<br>[l m n o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[g h i j] &nbsp [g h i k]<br>[a u c d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ]
    },
    "alph_1": {
      "shuffled_letters": [
        "l",
        "b"
      ],
      "shuffled_alphabet": [
        "a",
        "l",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "b",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z"
      ],
      "succ": [
        {
          "prompt": "[p q r s] &nbsp [p q r t]<br>[i j k b] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[i j k b] &nbsp [i j k m]<br>[a l c d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[a l c d] &nbsp [a l c e]<br>[m n o p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[l c d e] &nbsp [l c d f]<br>[e f g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[k b m n] &nbsp [k b m o]<br>[t u v w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[v w x y] &nbsp [v w x z]<br>[a l c d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[q r s t] &nbsp [q r s u]<br>[h i j k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[j k b m] &nbsp [j k b n]<br>[d e f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[k b m n] &nbsp [k b m o]<br>[b m n o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[d e f g] &nbsp [d e f h]<br>[a l c d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "pred": [
        {
          "prompt": "[i j k b] &nbsp [h j k b]<br>[k b m n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[g h i j] &nbsp [f h i j]<br>[l c d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[q r s t] &nbsp [p r s t]<br>[c d e f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[i j k b] &nbsp [h j k b]<br>[q r s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[c d e f] &nbsp [l d e f]<br>[i j k b] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[u v w x] &nbsp [t v w x]<br>[l c d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[t u v w] &nbsp [s u v w]<br>[k b m n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[i j k b] &nbsp [h j k b]<br>[v w x y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[l c d e] &nbsp [a c d e]<br>[r s t u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[i j k b] &nbsp [h j k b]<br>[e f g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "add_letter": [
        {
          "prompt": "[a l c d] &nbsp [a l c d e]<br>[r s t u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[h i j k] &nbsp [h i j k b]<br>[k b m n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[t u v w] &nbsp [t u v w x]<br>[h i j k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[d e f g] &nbsp [d e f g h]<br>[h i j k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[a l c d] &nbsp [a l c d e]<br>[l c d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[v w x y] &nbsp [v w x y z]<br>[i j k b] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[a l c d] &nbsp [a l c d e]<br>[v w x y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[e f g h] &nbsp [e f g h i]<br>[a l c d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[a l c d] &nbsp [a l c d e]<br>[b m n o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[i j k b] &nbsp [i j k b m]<br>[a l c d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "remove_redundant": [
        {
          "prompt": "[j k k b m n] &nbsp [j k b m n]<br>[s t u v w w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[l c d e f f] &nbsp [l c d e f]<br>[b b m n o p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[i i j k b m] &nbsp [i j k b m]<br>[a l c d d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[a a l c d e] &nbsp [a l c d e]<br>[v w x y y z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[r s s t u v] &nbsp [r s t u v]<br>[a l c d d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[a l c d d e] &nbsp [a l c d e]<br>[j k b m n n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[m n o p q q] &nbsp [m n o p q]<br>[l c d e f f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[a l c d d e] &nbsp [a l c d e]<br>[i j k k b m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[j k b m n n] &nbsp [j k b m n]<br>[k b m n n o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[i j k b m m] &nbsp [i j k b m]<br>[g g h i j k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "fix_alphabet": [
        {
          "prompt": "[g o i j k] &nbsp [g h i j k]<br>[a l c d i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[j k b y n] &nbsp [j k b m n]<br>[w p q r s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[t c d e f] &nbsp [l c d e f]<br>[i j k b d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[o p i r s] &nbsp [o p q r s]<br>[b m n o j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[a l q d e] &nbsp [a l c d e]<br>[e f p h i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[e f n h i] &nbsp [e f g h i]<br>[k l m n o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[l c d e i] &nbsp [l c d e f]<br>[k b h n o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[f r h i j] &nbsp [f g h i j]<br>[a l c d h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[a l c d j] &nbsp [a l c d e]<br>[u v w f y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[f l c d e] &nbsp [a l c d e]<br>[k s t u v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "sort": [
        {
          "prompt": "[o p r q s] &nbsp [o p q r s]<br>[k j b m n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[l e d c f] &nbsp [l c d e f]<br>[a l c e d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[w t u v s] &nbsp [s t u v w]<br>[c l a d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[j n b m k] &nbsp [j k b m n]<br>[k b m o n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[f g j i h] &nbsp [f g h i j]<br>[k b o n m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[a l d c e] &nbsp [a l c d e]<br>[m j k b i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[c l a d e] &nbsp [a l c d e]<br>[m j k b i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[p q r t s] &nbsp [p q r s t]<br>[e l c d a] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[q s r t u] &nbsp [q r s t u]<br>[n m b o p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[l f d e c] &nbsp [l c d e f]<br>[v s t u r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "attn": [
        {
          "prompt": "[a l c d] &nbsp [a l c e]<br>[u v w x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[f g h i] &nbsp [f g h j]<br>[b m n o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[a l c d] &nbsp [a l c e]<br>[p q r s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[a l c d] &nbsp [a l c e]<br>[h i j k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[a l c d] &nbsp [a l c e]<br>[f g h i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[m n o p] &nbsp [m n o q]<br>[b m n o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[k b m n] &nbsp [k b m o]<br>[l c d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[b m n o] &nbsp [b m n p]<br>[h i j k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[s t u v] &nbsp [s t u w]<br>[b m n o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[a l c d] &nbsp [a l c e]<br>[k b m n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ]
    },
    "alph_2": {
      "shuffled_letters": [
        "w",
        "s"
      ],
      "shuffled_alphabet": [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "w",
        "t",
        "u",
        "v",
        "s",
        "x",
        "y",
        "z"
      ],
      "succ": [
        {
          "prompt": "[w t u v] &nbsp [w t u s]<br>[a b c d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[w t u v] &nbsp [w t u s]<br>[v s x y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[f g h i] &nbsp [f g h j]<br>[r w t u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[o p q r] &nbsp [o p q w]<br>[e f g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[r w t u] &nbsp [r w t v]<br>[i j k l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[v s x y] &nbsp [v s x z]<br>[q r w t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[v s x y] &nbsp [v s x z]<br>[u v s x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[r w t u] &nbsp [r w t v]<br>[m n o p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[w t u v] &nbsp [w t u s]<br>[q r w t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[u v s x] &nbsp [u v s y]<br>[c d e f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "pred": [
        {
          "prompt": "[w t u v] &nbsp [r t u v]<br>[d e f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[w t u v] &nbsp [r t u v]<br>[p q r w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[r w t u] &nbsp [q w t u]<br>[t u v s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[q r w t] &nbsp [p r w t]<br>[k l m n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[t u v s] &nbsp [w u v s]<br>[s x y z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[k l m n] &nbsp [j l m n]<br>[r w t u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[r w t u] &nbsp [q w t u]<br>[s x y z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[t u v s] &nbsp [w u v s]<br>[k l m n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[s x y z] &nbsp [v x y z]<br>[v s x y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[p q r w] &nbsp [o q r w]<br>[g h i j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "add_letter": [
        {
          "prompt": "[v s x y] &nbsp [v s x y z]<br>[w t u v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[q r w t] &nbsp [q r w t u]<br>[o p q r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[l m n o] &nbsp [l m n o p]<br>[r w t u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[v s x y] &nbsp [v s x y z]<br>[o p q r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[u v s x] &nbsp [u v s x y]<br>[n o p q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[e f g h] &nbsp [e f g h i]<br>[q r w t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[t u v s] &nbsp [t u v s x]<br>[w t u v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[j k l m] &nbsp [j k l m n]<br>[v s x y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[q r w t] &nbsp [q r w t u]<br>[m n o p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[v s x y] &nbsp [v s x y z]<br>[r w t u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "remove_redundant": [
        {
          "prompt": "[m n o o p q] &nbsp [m n o p q]<br>[v s s x y z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[v s x y z z] &nbsp [v s x y z]<br>[d e f g h h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[o p q r r w] &nbsp [o p q r w]<br>[t u v s s x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[t u v v s x] &nbsp [t u v s x]<br>[n o p p q r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[p q q r w t] &nbsp [p q r w t]<br>[u v s s x y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[v s x y z z] &nbsp [v s x y z]<br>[f g g h i j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[b c d e e f] &nbsp [b c d e f]<br>[u v s x y y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[o o p q r w] &nbsp [o p q r w]<br>[k l m m n o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[q q r w t u] &nbsp [q r w t u]<br>[c d d e f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[t u v s s x] &nbsp [t u v s x]<br>[k k l m n o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "fix_alphabet": [
        {
          "prompt": "[o p q r h] &nbsp [o p q r w]<br>[w t u v k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[q r w z u] &nbsp [q r w t u]<br>[u v a x y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[x q r w t] &nbsp [p q r w t]<br>[w i u v s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[u v f x y] &nbsp [u v s x y]<br>[m n o c q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[t v s x y] &nbsp [u v s x y]<br>[p q x w t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[w n o p q] &nbsp [m n o p q]<br>[r w t u m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[t u v e x] &nbsp [t u v s x]<br>[k l m j o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[g h i t k] &nbsp [g h i j k]<br>[v s o y z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[l n o p q] &nbsp [m n o p q]<br>[v s x h z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[p q r w b] &nbsp [p q r w t]<br>[e f g h a] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "sort": [
        {
          "prompt": "[i j k m l] &nbsp [i j k l m]<br>[v s y x z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[t w u v s] &nbsp [w t u v s]<br>[t w r u v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[o q p r w] &nbsp [o p q r w]<br>[k j l m n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[w u t v s] &nbsp [w t u v s]<br>[e d f g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[n m o p q] &nbsp [m n o p q]<br>[w v u t s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[t q r w p] &nbsp [p q r w t]<br>[j k m l n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[w q r p t] &nbsp [p q r w t]<br>[t u v x s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[u r w t q] &nbsp [q r w t u]<br>[h k j i l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[c b d e f] &nbsp [b c d e f]<br>[q r u t w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[q r w u t] &nbsp [q r w t u]<br>[h g f i j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "attn": [
        {
          "prompt": "[t u v s] &nbsp [t u v x]<br>[w t u v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[p q r w] &nbsp [p q r t]<br>[r w t u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[u v s x] &nbsp [u v s y]<br>[v s x y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[r w t u] &nbsp [r w t v]<br>[w t u v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[a b c d] &nbsp [a b c e]<br>[w t u v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[v s x y] &nbsp [v s x z]<br>[q r w t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[b c d e] &nbsp [b c d f]<br>[u v s x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[w t u v] &nbsp [w t u s]<br>[g h i j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[h i j k] &nbsp [h i j l]<br>[v s x y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[w t u v] &nbsp [w t u s]<br>[f g h i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ]
    },
    "alph_3": {
      "shuffled_letters": [
        "y",
        "q"
      ],
      "shuffled_alphabet": [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "y",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "q",
        "z"
      ],
      "succ": [
        {
          "prompt": "[v w x q] &nbsp [v w x z]<br>[c d e f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[i j k l] &nbsp [i j k m]<br>[o p y r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[u v w x] &nbsp [u v w q]<br>[r s t u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[u v w x] &nbsp [u v w q]<br>[o p y r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[y r s t] &nbsp [y r s u]<br>[u v w x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[v w x q] &nbsp [v w x z]<br>[s t u v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[p y r s] &nbsp [p y r t]<br>[s t u v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[v w x q] &nbsp [v w x z]<br>[f g h i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[u v w x] &nbsp [u v w q]<br>[h i j k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[y r s t] &nbsp [y r s u]<br>[l m n o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "pred": [
        {
          "prompt": "[u v w x] &nbsp [t v w x]<br>[w x q z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[o p y r] &nbsp [n p y r]<br>[c d e f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[y r s t] &nbsp [p r s t]<br>[t u v w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[s t u v] &nbsp [r t u v]<br>[p y r s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[r s t u] &nbsp [y s t u]<br>[w x q z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[j k l m] &nbsp [i k l m]<br>[w x q z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[w x q z] &nbsp [v x q z]<br>[s t u v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[p y r s] &nbsp [o y r s]<br>[t u v w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[w x q z] &nbsp [v x q z]<br>[u v w x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[r s t u] &nbsp [y s t u]<br>[y r s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "add_letter": [
        {
          "prompt": "[o p y r] &nbsp [o p y r s]<br>[p y r s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[n o p y] &nbsp [n o p y r]<br>[k l m n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[y r s t] &nbsp [y r s t u]<br>[j k l m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[k l m n] &nbsp [k l m n o]<br>[v w x q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[y r s t] &nbsp [y r s t u]<br>[o p y r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[g h i j] &nbsp [g h i j k]<br>[v w x q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[m n o p] &nbsp [m n o p y]<br>[t u v w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[c d e f] &nbsp [c d e f g]<br>[y r s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[p y r s] &nbsp [p y r s t]<br>[f g h i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[m n o p] &nbsp [m n o p y]<br>[y r s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "remove_redundant": [
        {
          "prompt": "[f g h i i j] &nbsp [f g h i j]<br>[u v w x q q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[g h i j j k] &nbsp [g h i j k]<br>[v w x q z z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[p p y r s t] &nbsp [p y r s t]<br>[y r r s t u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[p y r r s t] &nbsp [p y r s t]<br>[e e f g h i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[m n o p y y] &nbsp [m n o p y]<br>[h i j k l l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[p y r r s t] &nbsp [p y r s t]<br>[t u v w w x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[v v w x q z] &nbsp [v w x q z]<br>[b c d d e f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[v w x q q z] &nbsp [v w x q z]<br>[m n o p y y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[y y r s t u] &nbsp [y r s t u]<br>[f g h i j j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[v w x q z z] &nbsp [v w x q z]<br>[d e f f g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "fix_alphabet": [
        {
          "prompt": "[a t u v w] &nbsp [s t u v w]<br>[v w x u z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[u f w x q] &nbsp [u v w x q]<br>[i z k l m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[v l x q z] &nbsp [v w x q z]<br>[m z o p y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[l m n o d] &nbsp [l m n o p]<br>[y r s o u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[h i j k m] &nbsp [h i j k l]<br>[v w x o z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[s t u v c] &nbsp [s t u v w]<br>[a w x q z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[t u v n x] &nbsp [t u v w x]<br>[p v w x q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[v w x q y] &nbsp [v w x q z]<br>[n o p m r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[h y j k l] &nbsp [h i j k l]<br>[x y r s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[l p y r s] &nbsp [o p y r s]<br>[s n o p y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "sort": [
        {
          "prompt": "[e f h g i] &nbsp [e f g h i]<br>[x w v q z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[v q x w z] &nbsp [v w x q z]<br>[d h f g e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[x w v q z] &nbsp [v w x q z]<br>[d b c a e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[j m l k n] &nbsp [j k l m n]<br>[n r p y o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[o p s r y] &nbsp [o p y r s]<br>[v q x w z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[c e d f g] &nbsp [c d e f g]<br>[z w x q v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[y r u t s] &nbsp [y r s t u]<br>[f i h g j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[f g i h j] &nbsp [f g h i j]<br>[q w x v z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[g h i k j] &nbsp [g h i j k]<br>[y r s u t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[n y p o r] &nbsp [n o p y r]<br>[u r s t y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "attn": [
        {
          "prompt": "[t u v w] &nbsp [t u v x]<br>[v w x q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[p y r s] &nbsp [p y r t]<br>[v w x q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[g h i j] &nbsp [g h i k]<br>[v w x q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[r s t u] &nbsp [r s t v]<br>[o p y r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[r s t u] &nbsp [r s t v]<br>[y r s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[u v w x] &nbsp [u v w q]<br>[l m n o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[m n o p] &nbsp [m n o y]<br>[v w x q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[i j k l] &nbsp [i j k m]<br>[v w x q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[b c d e] &nbsp [b c d f]<br>[y r s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[v w x q] &nbsp [v w x z]<br>[j k l m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ]
    },
    "alph_4": {
      "shuffled_letters": [
        "f",
        "c"
      ],
      "shuffled_alphabet": [
        "a",
        "b",
        "f",
        "d",
        "e",
        "c",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z"
      ],
      "succ": [
        {
          "prompt": "[a b f d] &nbsp [a b f e]<br>[c g h i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[a b f d] &nbsp [a b f e]<br>[f d e c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[r s t u] &nbsp [r s t v]<br>[f d e c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[d e c g] &nbsp [d e c h]<br>[f d e c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[l m n o] &nbsp [l m n p]<br>[a b f d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[m n o p] &nbsp [m n o q]<br>[c g h i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[s t u v] &nbsp [s t u w]<br>[f d e c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[f d e c] &nbsp [f d e g]<br>[m n o p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[d e c g] &nbsp [d e c h]<br>[p q r s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[f d e c] &nbsp [f d e g]<br>[u v w x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "pred": [
        {
          "prompt": "[b f d e] &nbsp [a f d e]<br>[k l m n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[f d e c] &nbsp [b d e c]<br>[l m n o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[f d e c] &nbsp [b d e c]<br>[d e c g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[e c g h] &nbsp [d c g h]<br>[f d e c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[d e c g] &nbsp [f e c g]<br>[j k l m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[f d e c] &nbsp [b d e c]<br>[b f d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[i j k l] &nbsp [h j k l]<br>[b f d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[f d e c] &nbsp [b d e c]<br>[t u v w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[f d e c] &nbsp [b d e c]<br>[i j k l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[e c g h] &nbsp [d c g h]<br>[r s t u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "add_letter": [
        {
          "prompt": "[b f d e] &nbsp [b f d e c]<br>[r s t u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[u v w x] &nbsp [u v w x y]<br>[d e c g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[j k l m] &nbsp [j k l m n]<br>[b f d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[b f d e] &nbsp [b f d e c]<br>[c g h i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[h i j k] &nbsp [h i j k l]<br>[f d e c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[f d e c] &nbsp [f d e c g]<br>[a b f d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[a b f d] &nbsp [a b f d e]<br>[i j k l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[j k l m] &nbsp [j k l m n]<br>[a b f d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[h i j k] &nbsp [h i j k l]<br>[c g h i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[s t u v] &nbsp [s t u v w]<br>[a b f d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "remove_redundant": [
        {
          "prompt": "[p p q r s t] &nbsp [p q r s t]<br>[a b b f d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[e c c g h i] &nbsp [e c g h i]<br>[b b f d e c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[t t u v w x] &nbsp [t u v w x]<br>[c g h h i j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[e e c g h i] &nbsp [e c g h i]<br>[a b f d d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[g g h i j k] &nbsp [g h i j k]<br>[b f d e c c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[n o o p q r] &nbsp [n o p q r]<br>[f d e c c g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[q r s s t u] &nbsp [q r s t u]<br>[a b f f d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[a b f d d e] &nbsp [a b f d e]<br>[n o o p q r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[l m n n o p] &nbsp [l m n o p]<br>[a b f d d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[c c g h i j] &nbsp [c g h i j]<br>[q q r s t u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "fix_alphabet": [
        {
          "prompt": "[b f l e c] &nbsp [b f d e c]<br>[d e c g p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[u a w x y] &nbsp [u v w x y]<br>[m c g h i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[f d e c w] &nbsp [f d e c g]<br>[n b f d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[a b w d e] &nbsp [a b f d e]<br>[k y m n o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[c g h i t] &nbsp [c g h i j]<br>[q u v w x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[s t u v m] &nbsp [s t u v w]<br>[n f d e c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[n g h i j] &nbsp [c g h i j]<br>[d e c g a] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[f d s c g] &nbsp [f d e c g]<br>[i r s t u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[b f d j c] &nbsp [b f d e c]<br>[o p g r s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[f d e c p] &nbsp [f d e c g]<br>[p q d s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "sort": [
        {
          "prompt": "[e g c h i] &nbsp [e c g h i]<br>[f c e d g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[t u v x w] &nbsp [t u v w x]<br>[f b d e c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[d b f a e] &nbsp [a b f d e]<br>[j i h k l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[b c d e f] &nbsp [b f d e c]<br>[c d e f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[f b d e c] &nbsp [b f d e c]<br>[u w v x y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[f b d e c] &nbsp [b f d e c]<br>[n m o p q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[a b d f e] &nbsp [a b f d e]<br>[q r s u t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[c d e f g] &nbsp [f d e c g]<br>[d f b e c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[e c h g i] &nbsp [e c g h i]<br>[k j l m n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[q p o r s] &nbsp [o p q r s]<br>[e b f d a] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "attn": [
        {
          "prompt": "[a b f d] &nbsp [a b f e]<br>[o p q r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[a b f d] &nbsp [a b f e]<br>[c g h i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[e c g h] &nbsp [e c g i]<br>[a b f d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[n o p q] &nbsp [n o p r]<br>[a b f d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[f d e c] &nbsp [f d e g]<br>[d e c g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[b f d e] &nbsp [b f d c]<br>[o p q r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[a b f d] &nbsp [a b f e]<br>[f d e c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[r s t u] &nbsp [r s t v]<br>[f d e c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[f d e c] &nbsp [f d e g]<br>[q r s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[h i j k] &nbsp [h i j l]<br>[d e c g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ]
    },
    "alph_5": {
      "shuffled_letters": [
        "l",
        "f"
      ],
      "shuffled_alphabet": [
        "a",
        "b",
        "c",
        "d",
        "e",
        "l",
        "g",
        "h",
        "i",
        "j",
        "k",
        "f",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z"
      ],
      "succ": [
        {
          "prompt": "[k f m n] &nbsp [k f m o]<br>[j k f m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[b c d e] &nbsp [b c d l]<br>[h i j k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[f m n o] &nbsp [f m n p]<br>[u v w x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[i j k f] &nbsp [i j k m]<br>[r s t u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[i j k f] &nbsp [i j k m]<br>[a b c d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[h i j k] &nbsp [h i j f]<br>[p q r s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[b c d e] &nbsp [b c d l]<br>[f m n o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[c d e l] &nbsp [c d e g]<br>[a b c d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[t u v w] &nbsp [t u v x]<br>[k f m n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[i j k f] &nbsp [i j k m]<br>[k f m n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "pred": [
        {
          "prompt": "[s t u v] &nbsp [r t u v]<br>[c d e l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[l g h i] &nbsp [e g h i]<br>[v w x y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[c d e l] &nbsp [b d e l]<br>[m n o p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[k f m n] &nbsp [j f m n]<br>[i j k f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[d e l g] &nbsp [c e l g]<br>[k f m n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[f m n o] &nbsp [k m n o]<br>[n o p q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[l g h i] &nbsp [e g h i]<br>[f m n o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[k f m n] &nbsp [j f m n]<br>[u v w x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[t u v w] &nbsp [s u v w]<br>[c d e l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[p q r s] &nbsp [o q r s]<br>[c d e l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "add_letter": [
        {
          "prompt": "[h i j k] &nbsp [h i j k f]<br>[e l g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[c d e l] &nbsp [c d e l g]<br>[d e l g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[h i j k] &nbsp [h i j k f]<br>[c d e l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[r s t u] &nbsp [r s t u v]<br>[k f m n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[l g h i] &nbsp [l g h i j]<br>[c d e l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[r s t u] &nbsp [r s t u v]<br>[c d e l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[f m n o] &nbsp [f m n o p]<br>[c d e l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[i j k f] &nbsp [i j k f m]<br>[k f m n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[j k f m] &nbsp [j k f m n]<br>[k f m n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[l g h i] &nbsp [l g h i j]<br>[j k f m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "remove_redundant": [
        {
          "prompt": "[o p q q r s] &nbsp [o p q r s]<br>[d e l g h h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[s t t u v w] &nbsp [s t u v w]<br>[d e l l g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[i i j k f m] &nbsp [i j k f m]<br>[j k f f m n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[j k f f m n] &nbsp [j k f m n]<br>[d e l l g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[e l g h i i] &nbsp [e l g h i]<br>[j k f f m n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[n n o p q r] &nbsp [n o p q r]<br>[f m n o p p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[d e l g h h] &nbsp [d e l g h]<br>[c d d e l g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[h i j j k f] &nbsp [h i j k f]<br>[j j k f m n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[h i i j k f] &nbsp [h i j k f]<br>[b c d d e l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[a b c c d e] &nbsp [a b c d e]<br>[j k k f m n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "fix_alphabet": [
        {
          "prompt": "[g v i j k] &nbsp [g h i j k]<br>[c d e l u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[g h i r k] &nbsp [g h i j k]<br>[t j k f m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[s b u v w] &nbsp [s t u v w]<br>[k f m n e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[e l g h b] &nbsp [e l g h i]<br>[o p q z s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[f m v o p] &nbsp [f m n o p]<br>[j o f m n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[i u k f m] &nbsp [i j k f m]<br>[k r s t u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[j k f m x] &nbsp [j k f m n]<br>[v j k f m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[b c t e l] &nbsp [b c d e l]<br>[m e l g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[w g h i j] &nbsp [l g h i j]<br>[k f r n o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[l g h f j] &nbsp [l g h i j]<br>[j k c m n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "sort": [
        {
          "prompt": "[e l i h g] &nbsp [e l g h i]<br>[h j i k f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[h k j i f] &nbsp [h i j k f]<br>[i k j f m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[i g h l j] &nbsp [l g h i j]<br>[d l e g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[h j i k f] &nbsp [h i j k f]<br>[b l d e c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[q p o r s] &nbsp [o p q r s]<br>[l i h g j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[m j k f i] &nbsp [i j k f m]<br>[f m p o n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[j n f m k] &nbsp [j k f m n]<br>[k h i j g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[o r q p s] &nbsp [o p q r s]<br>[e i g h l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[g h k j i] &nbsp [g h i j k]<br>[i f k j m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[f m p o n] &nbsp [f m n o p]<br>[h f j k i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "attn": [
        {
          "prompt": "[k f m n] &nbsp [k f m o]<br>[f m n o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[d e l g] &nbsp [d e l h]<br>[t u v w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[g h i j] &nbsp [g h i k]<br>[b c d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[e l g h] &nbsp [e l g i]<br>[j k f m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[f m n o] &nbsp [f m n p]<br>[t u v w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[j k f m] &nbsp [j k f n]<br>[p q r s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[j k f m] &nbsp [j k f n]<br>[i j k f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[b c d e] &nbsp [b c d l]<br>[c d e l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[b c d e] &nbsp [b c d l]<br>[d e l g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[i j k f] &nbsp [i j k m]<br>[f m n o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ]
    },
    "alph_6": {
      "shuffled_letters": [
        "m",
        "e"
      ],
      "shuffled_alphabet": [
        "a",
        "b",
        "c",
        "d",
        "m",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "e",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z"
      ],
      "succ": [
        {
          "prompt": "[h i j k] &nbsp [h i j l]<br>[k l e n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[m f g h] &nbsp [m f g i]<br>[d m f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[e n o p] &nbsp [e n o q]<br>[p q r s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[c d m f] &nbsp [c d m g]<br>[d m f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[n o p q] &nbsp [n o p r]<br>[e n o p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[e n o p] &nbsp [e n o q]<br>[a b c d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[k l e n] &nbsp [k l e o]<br>[a b c d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[a b c d] &nbsp [a b c m]<br>[l e n o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[j k l e] &nbsp [j k l n]<br>[i j k l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[a b c d] &nbsp [a b c m]<br>[k l e n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "pred": [
        {
          "prompt": "[v w x y] &nbsp [u w x y]<br>[d m f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[k l e n] &nbsp [j l e n]<br>[f g h i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[o p q r] &nbsp [n p q r]<br>[k l e n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[l e n o] &nbsp [k e n o]<br>[c d m f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[d m f g] &nbsp [c m f g]<br>[k l e n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[j k l e] &nbsp [i k l e]<br>[n o p q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[p q r s] &nbsp [o q r s]<br>[n o p q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[j k l e] &nbsp [i k l e]<br>[b c d m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[b c d m] &nbsp [a c d m]<br>[d m f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[w x y z] &nbsp [v x y z]<br>[e n o p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "add_letter": [
        {
          "prompt": "[d m f g] &nbsp [d m f g h]<br>[v w x y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[h i j k] &nbsp [h i j k l]<br>[c d m f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[k l e n] &nbsp [k l e n o]<br>[d m f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[b c d m] &nbsp [b c d m f]<br>[m f g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[b c d m] &nbsp [b c d m f]<br>[j k l e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[i j k l] &nbsp [i j k l e]<br>[b c d m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[p q r s] &nbsp [p q r s t]<br>[d m f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[a b c d] &nbsp [a b c d m]<br>[e n o p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[e n o p] &nbsp [e n o p q]<br>[f g h i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[b c d m] &nbsp [b c d m f]<br>[h i j k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "remove_redundant": [
        {
          "prompt": "[b c d m f f] &nbsp [b c d m f]<br>[k l e n n o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[a b c d m m] &nbsp [a b c d m]<br>[c c d m f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[b c d d m f] &nbsp [b c d m f]<br>[u v w x y y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[f f g h i j] &nbsp [f g h i j]<br>[m f f g h i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[a b c d d m] &nbsp [a b c d m]<br>[b c c d m f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[h i j k l l] &nbsp [h i j k l]<br>[m f g h i i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[m f g h i i] &nbsp [m f g h i]<br>[d m f g h h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[l l e n o p] &nbsp [l e n o p]<br>[b c d d m f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[k l e e n o] &nbsp [k l e n o]<br>[t u u v w x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[m f g g h i] &nbsp [m f g h i]<br>[l e n n o p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "fix_alphabet": [
        {
          "prompt": "[j k l g n] &nbsp [j k l e n]<br>[g h m j k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[k u e n o] &nbsp [k l e n o]<br>[a b c u m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[c d v f g] &nbsp [c d m f g]<br>[e d o p q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[e n w p q] &nbsp [e n o p q]<br>[m f g h j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[a b c q m] &nbsp [a b c d m]<br>[d m f l h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[q r h t u] &nbsp [q r s t u]<br>[l e n i p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[i j z l e] &nbsp [i j k l e]<br>[f n o p q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[l j n o p] &nbsp [l e n o p]<br>[k l e p o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[c p m f g] &nbsp [c d m f g]<br>[l a n o p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[m v g h i] &nbsp [m f g h i]<br>[f g l i j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "sort": [
        {
          "prompt": "[i f g h m] &nbsp [m f g h i]<br>[i j e l k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[a c b d m] &nbsp [a b c d m]<br>[w v u x y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[b m d c f] &nbsp [b c d m f]<br>[x w v y z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[d c b m f] &nbsp [b c d m f]<br>[l e o n p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[i h g j k] &nbsp [g h i j k]<br>[c f m d g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[k l n e o] &nbsp [k l e n o]<br>[k h i j g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[d f m g h] &nbsp [d m f g h]<br>[i j k e l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[n l e k o] &nbsp [k l e n o]<br>[l e p o n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[c b a d m] &nbsp [a b c d m]<br>[c f m d g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[a m c d b] &nbsp [a b c d m]<br>[l i j k h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "attn": [
        {
          "prompt": "[i j k l] &nbsp [i j k e]<br>[m f g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[o p q r] &nbsp [o p q s]<br>[m f g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[v w x y] &nbsp [v w x z]<br>[c d m f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[m f g h] &nbsp [m f g i]<br>[c d m f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[o p q r] &nbsp [o p q s]<br>[i j k l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[i j k l] &nbsp [i j k e]<br>[c d m f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[f g h i] &nbsp [f g h j]<br>[m f g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[f g h i] &nbsp [f g h j]<br>[k l e n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[m f g h] &nbsp [m f g i]<br>[q r s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[e n o p] &nbsp [e n o q]<br>[v w x y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ]
    }
  },
  "np_5": {
    "alph_0": {
      "shuffled_letters": [
        "n",
        "b",
        "q",
        "u",
        "o"
      ],
      "shuffled_alphabet": [
        "a",
        "n",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "b",
        "q",
        "p",
        "u",
        "r",
        "s",
        "t",
        "o",
        "v",
        "w",
        "x",
        "y",
        "z"
      ],
      "succ": [
        {
          "prompt": "[f g h i] &nbsp [f g h j]<br>[k l m b] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[j k l m] &nbsp [j k l b]<br>[p u r s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[m b q p] &nbsp [m b q u]<br>[k l m b] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[k l m b] &nbsp [k l m q]<br>[a n c d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[p u r s] &nbsp [p u r t]<br>[t o v w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[q p u r] &nbsp [q p u s]<br>[r s t o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[k l m b] &nbsp [k l m q]<br>[q p u r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[g h i j] &nbsp [g h i k]<br>[a n c d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[p u r s] &nbsp [p u r t]<br>[u r s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[q p u r] &nbsp [q p u s]<br>[j k l m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "pred": [
        {
          "prompt": "[m b q p] &nbsp [l b q p]<br>[e f g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[p u r s] &nbsp [q u r s]<br>[c d e f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[q p u r] &nbsp [b p u r]<br>[v w x y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[m b q p] &nbsp [l b q p]<br>[s t o v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[l m b q] &nbsp [k m b q]<br>[v w x y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[m b q p] &nbsp [l b q p]<br>[p u r s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[v w x y] &nbsp [o w x y]<br>[p u r s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[b q p u] &nbsp [m q p u]<br>[q p u r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[b q p u] &nbsp [m q p u]<br>[s t o v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[u r s t] &nbsp [p r s t]<br>[q p u r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "add_letter": [
        {
          "prompt": "[u r s t] &nbsp [u r s t o]<br>[g h i j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[u r s t] &nbsp [u r s t o]<br>[t o v w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[g h i j] &nbsp [g h i j k]<br>[b q p u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[t o v w] &nbsp [t o v w x]<br>[a n c d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[l m b q] &nbsp [l m b q p]<br>[u r s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[t o v w] &nbsp [t o v w x]<br>[j k l m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[b q p u] &nbsp [b q p u r]<br>[h i j k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[l m b q] &nbsp [l m b q p]<br>[a n c d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[p u r s] &nbsp [p u r s t]<br>[q p u r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[b q p u] &nbsp [b q p u r]<br>[a n c d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "remove_redundant": [
        {
          "prompt": "[j j k l m b] &nbsp [j k l m b]<br>[g h i i j k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[k l m m b q] &nbsp [k l m b q]<br>[b b q p u r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[a n c d e e] &nbsp [a n c d e]<br>[n c c d e f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[u r r s t o] &nbsp [u r s t o]<br>[v w x y y z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[h i j j k l] &nbsp [h i j k l]<br>[a n c c d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[q p u r s s] &nbsp [q p u r s]<br>[r s t t o v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[a a n c d e] &nbsp [a n c d e]<br>[q p u u r s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[t o v w w x] &nbsp [t o v w x]<br>[k l m b q q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[f g g h i j] &nbsp [f g h i j]<br>[a a n c d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[a a n c d e] &nbsp [a n c d e]<br>[c d d e f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "fix_alphabet": [
        {
          "prompt": "[q p u r o] &nbsp [q p u r s]<br>[j k l m c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[y g h i j] &nbsp [f g h i j]<br>[p u r v t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[b n p u r] &nbsp [b q p u r]<br>[e f r h i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[e z g h i] &nbsp [e f g h i]<br>[q p n r s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[u r s k o] &nbsp [u r s t o]<br>[l h b q p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[e b g h i] &nbsp [e f g h i]<br>[o a w x y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[s t i v w] &nbsp [s t o v w]<br>[d q f g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[a q p u r] &nbsp [b q p u r]<br>[o v j x y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[p u r w t] &nbsp [p u r s t]<br>[a n c d p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[e f g y i] &nbsp [e f g h i]<br>[a o v w x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "sort": [
        {
          "prompt": "[e c d n f] &nbsp [n c d e f]<br>[u b q p m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[g e f d h] &nbsp [d e f g h]<br>[m q b p u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[a c n d e] &nbsp [a n c d e]<br>[q p s r u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[r u p s t] &nbsp [p u r s t]<br>[i j l k m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[f c d e n] &nbsp [n c d e f]<br>[h e f g d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[b u p q r] &nbsp [b q p u r]<br>[w o v t x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[o v y x w] &nbsp [o v w x y]<br>[u t s r o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[n c f e d] &nbsp [n c d e f]<br>[a n d c e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[b q p r u] &nbsp [b q p u r]<br>[s t w v o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[m q b p u] &nbsp [m b q p u]<br>[b u p q r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "attn": [
        {
          "prompt": "[u r s t] &nbsp [u r s o]<br>[b q p u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[k l m b] &nbsp [k l m q]<br>[b q p u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[v w x y] &nbsp [v w x z]<br>[b q p u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[a n c d] &nbsp [a n c e]<br>[p u r s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[q p u r] &nbsp [q p u s]<br>[n c d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[m b q p] &nbsp [m b q u]<br>[a n c d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[a n c d] &nbsp [a n c e]<br>[s t o v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[u r s t] &nbsp [u r s o]<br>[k l m b] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[a n c d] &nbsp [a n c e]<br>[i j k l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[p u r s] &nbsp [p u r t]<br>[r s t o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ]
    },
    "alph_1": {
      "shuffled_letters": [
        "o",
        "r",
        "l",
        "x",
        "e"
      ],
      "shuffled_alphabet": [
        "a",
        "b",
        "c",
        "d",
        "o",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "r",
        "m",
        "n",
        "l",
        "p",
        "q",
        "x",
        "s",
        "t",
        "u",
        "v",
        "w",
        "e",
        "y",
        "z"
      ],
      "succ": [
        {
          "prompt": "[d o f g] &nbsp [d o f h]<br>[q x s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[k r m n] &nbsp [k r m l]<br>[c d o f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[n l p q] &nbsp [n l p x]<br>[q x s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[j k r m] &nbsp [j k r n]<br>[b c d o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[j k r m] &nbsp [j k r n]<br>[c d o f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[t u v w] &nbsp [t u v e]<br>[v w e y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[h i j k] &nbsp [h i j r]<br>[a b c d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[h i j k] &nbsp [h i j r]<br>[p q x s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[b c d o] &nbsp [b c d f]<br>[v w e y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[b c d o] &nbsp [b c d f]<br>[t u v w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "pred": [
        {
          "prompt": "[p q x s] &nbsp [l q x s]<br>[d o f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[u v w e] &nbsp [t v w e]<br>[x s t u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[b c d o] &nbsp [a c d o]<br>[v w e y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[w e y z] &nbsp [v e y z]<br>[s t u v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[i j k r] &nbsp [h j k r]<br>[t u v w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[t u v w] &nbsp [s u v w]<br>[x s t u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[f g h i] &nbsp [o g h i]<br>[l p q x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[r m n l] &nbsp [k m n l]<br>[q x s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[m n l p] &nbsp [r n l p]<br>[k r m n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[m n l p] &nbsp [r n l p]<br>[b c d o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "add_letter": [
        {
          "prompt": "[a b c d] &nbsp [a b c d o]<br>[g h i j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[o f g h] &nbsp [o f g h i]<br>[m n l p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[i j k r] &nbsp [i j k r m]<br>[l p q x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[j k r m] &nbsp [j k r m n]<br>[l p q x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[l p q x] &nbsp [l p q x s]<br>[f g h i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[k r m n] &nbsp [k r m n l]<br>[o f g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[d o f g] &nbsp [d o f g h]<br>[p q x s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[g h i j] &nbsp [g h i j k]<br>[i j k r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[i j k r] &nbsp [i j k r m]<br>[m n l p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[b c d o] &nbsp [b c d o f]<br>[s t u v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "remove_redundant": [
        {
          "prompt": "[q q x s t u] &nbsp [q x s t u]<br>[s t u v v w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[x s s t u v] &nbsp [x s t u v]<br>[u v w e e y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[a b c d d o] &nbsp [a b c d o]<br>[j k r m n n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[m n l p q q] &nbsp [m n l p q]<br>[o f g h i i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[k r m n l l] &nbsp [k r m n l]<br>[x s t t u v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[j k k r m n] &nbsp [j k r m n]<br>[k k r m n l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[i j k r m m] &nbsp [i j k r m]<br>[v w e y y z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[x s t u u v] &nbsp [x s t u v]<br>[r r m n l p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[p q q x s t] &nbsp [p q x s t]<br>[k r m n l l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[i j k r r m] &nbsp [i j k r m]<br>[r m n n l p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "fix_alphabet": [
        {
          "prompt": "[e i j k r] &nbsp [h i j k r]<br>[a b c z o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[f s t u v] &nbsp [x s t u v]<br>[r m k l p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[d o b g h] &nbsp [d o f g h]<br>[r m i l p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[l p q x f] &nbsp [l p q x s]<br>[m v l p q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[z t u v w] &nbsp [s t u v w]<br>[v x s t u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[p q x m t] &nbsp [p q x s t]<br>[m n b p q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[k y m n l] &nbsp [k r m n l]<br>[v w p y z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[q x s t e] &nbsp [q x s t u]<br>[d o f g r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[a w e y z] &nbsp [v w e y z]<br>[p g h i j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[l p q z s] &nbsp [l p q x s]<br>[p q x m t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "sort": [
        {
          "prompt": "[o f i h g] &nbsp [o f g h i]<br>[v u w e y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[h o f g d] &nbsp [d o f g h]<br>[y v w e u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[y v w e u] &nbsp [u v w e y]<br>[k m r n l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[k j i r m] &nbsp [i j k r m]<br>[l x q p s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[x s t v u] &nbsp [x s t u v]<br>[h j i k r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[s p q x l] &nbsp [l p q x s]<br>[h g f i j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[k m r n l] &nbsp [k r m n l]<br>[d f o g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[t x s q u] &nbsp [q x s t u]<br>[b f d o c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[t s x u v] &nbsp [x s t u v]<br>[m q l p n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[s t w v u] &nbsp [s t u v w]<br>[m r n l p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "attn": [
        {
          "prompt": "[n l p q] &nbsp [n l p x]<br>[i j k r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[f g h i] &nbsp [f g h j]<br>[a b c d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[c d o f] &nbsp [c d o g]<br>[t u v w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[k r m n] &nbsp [k r m l]<br>[g h i j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[v w e y] &nbsp [v w e z]<br>[d o f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[q x s t] &nbsp [q x s u]<br>[k r m n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[k r m n] &nbsp [k r m l]<br>[u v w e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[f g h i] &nbsp [f g h j]<br>[v w e y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[a b c d] &nbsp [a b c o]<br>[j k r m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[k r m n] &nbsp [k r m l]<br>[h i j k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ]
    },
    "alph_2": {
      "shuffled_letters": [
        "q",
        "o",
        "a",
        "h",
        "m"
      ],
      "shuffled_alphabet": [
        "q",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "o",
        "i",
        "j",
        "k",
        "l",
        "a",
        "n",
        "h",
        "p",
        "m",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z"
      ],
      "succ": [
        {
          "prompt": "[g o i j] &nbsp [g o i k]<br>[v w x y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[a n h p] &nbsp [a n h m]<br>[c d e f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[i j k l] &nbsp [i j k a]<br>[h p m r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[l a n h] &nbsp [l a n p]<br>[p m r s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[a n h p] &nbsp [a n h m]<br>[n h p m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[i j k l] &nbsp [i j k a]<br>[j k l a] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[o i j k] &nbsp [o i j l]<br>[h p m r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[m r s t] &nbsp [m r s u]<br>[j k l a] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[v w x y] &nbsp [v w x z]<br>[a n h p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[i j k l] &nbsp [i j k a]<br>[f g o i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "pred": [
        {
          "prompt": "[i j k l] &nbsp [o j k l]<br>[t u v w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[l a n h] &nbsp [k a n h]<br>[p m r s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[a n h p] &nbsp [l n h p]<br>[b c d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[h p m r] &nbsp [n p m r]<br>[l a n h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[k l a n] &nbsp [j l a n]<br>[g o i j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[g o i j] &nbsp [f o i j]<br>[b c d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[h p m r] &nbsp [n p m r]<br>[d e f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[i j k l] &nbsp [o j k l]<br>[o i j k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[e f g o] &nbsp [d f g o]<br>[b c d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[g o i j] &nbsp [f o i j]<br>[h p m r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "add_letter": [
        {
          "prompt": "[a n h p] &nbsp [a n h p m]<br>[g o i j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[l a n h] &nbsp [l a n h p]<br>[q b c d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[f g o i] &nbsp [f g o i j]<br>[l a n h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[c d e f] &nbsp [c d e f g]<br>[e f g o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[h p m r] &nbsp [h p m r s]<br>[d e f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[f g o i] &nbsp [f g o i j]<br>[p m r s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[b c d e] &nbsp [b c d e f]<br>[n h p m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[q b c d] &nbsp [q b c d e]<br>[m r s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[l a n h] &nbsp [l a n h p]<br>[n h p m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[a n h p] &nbsp [a n h p m]<br>[q b c d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "remove_redundant": [
        {
          "prompt": "[n h p p m r] &nbsp [n h p m r]<br>[h p m m r s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[e f g g o i] &nbsp [e f g o i]<br>[h p p m r s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[b c d e f f] &nbsp [b c d e f]<br>[q b c d e e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[q q b c d e] &nbsp [q b c d e]<br>[k l a a n h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[q q b c d e] &nbsp [q b c d e]<br>[c d d e f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[e f g o o i] &nbsp [e f g o i]<br>[k k l a n h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[a a n h p m] &nbsp [a n h p m]<br>[s t u u v w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[k l a n h h] &nbsp [k l a n h]<br>[o i j k k l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[p m r r s t] &nbsp [p m r s t]<br>[n h h p m r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[m r s s t u] &nbsp [m r s t u]<br>[q b c d d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "fix_alphabet": [
        {
          "prompt": "[n h p w r] &nbsp [n h p m r]<br>[d e q g o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[x h p m r] &nbsp [n h p m r]<br>[s i u v w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[a n j p m] &nbsp [a n h p m]<br>[n h p c r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[u z w x y] &nbsp [u v w x y]<br>[j k z a n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[m r q t u] &nbsp [m r s t u]<br>[a n w p m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[q s c d e] &nbsp [q b c d e]<br>[a n h p z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[i j m l a] &nbsp [i j k l a]<br>[c i j k l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[p m r l t] &nbsp [p m r s t]<br>[o f j k l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[o i j k q] &nbsp [o i j k l]<br>[m s t u v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[l a n q p] &nbsp [l a n h p]<br>[m r s n u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "sort": [
        {
          "prompt": "[m h p n r] &nbsp [n h p m r]<br>[t u v x w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[e b c d q] &nbsp [q b c d e]<br>[m t s r u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[c b d e f] &nbsp [b c d e f]<br>[q b e d c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[k l a h n] &nbsp [k l a n h]<br>[o g f i j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[u t v w x] &nbsp [t u v w x]<br>[c b q d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[d o f g e] &nbsp [d e f g o]<br>[b d c e f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[r m s t u] &nbsp [m r s t u]<br>[q b d c e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[l i j k o] &nbsp [o i j k l]<br>[p t r s m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[p h m r s] &nbsp [h p m r s]<br>[j n l a k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[i o j k l] &nbsp [o i j k l]<br>[g f o i j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "attn": [
        {
          "prompt": "[d e f g] &nbsp [d e f o]<br>[v w x y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[q b c d] &nbsp [q b c e]<br>[h p m r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[h p m r] &nbsp [h p m s]<br>[i j k l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[q b c d] &nbsp [q b c e]<br>[n h p m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[l a n h] &nbsp [l a n p]<br>[p m r s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[o i j k] &nbsp [o i j l]<br>[m r s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[g o i j] &nbsp [g o i k]<br>[e f g o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[q b c d] &nbsp [q b c e]<br>[r s t u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[i j k l] &nbsp [i j k a]<br>[b c d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[q b c d] &nbsp [q b c e]<br>[f g o i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ]
    },
    "alph_3": {
      "shuffled_letters": [
        "g",
        "q",
        "z",
        "n",
        "b"
      ],
      "shuffled_alphabet": [
        "a",
        "g",
        "c",
        "d",
        "e",
        "f",
        "q",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "z",
        "o",
        "p",
        "n",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "b"
      ],
      "succ": [
        {
          "prompt": "[m z o p] &nbsp [m z o n]<br>[t u v w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[s t u v] &nbsp [s t u w]<br>[a g c d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[z o p n] &nbsp [z o p r]<br>[d e f q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[m z o p] &nbsp [m z o n]<br>[g c d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[n r s t] &nbsp [n r s u]<br>[z o p n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[p n r s] &nbsp [p n r t]<br>[u v w x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[j k l m] &nbsp [j k l z]<br>[p n r s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[u v w x] &nbsp [u v w y]<br>[v w x y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[j k l m] &nbsp [j k l z]<br>[a g c d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[f q h i] &nbsp [f q h j]<br>[v w x y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "pred": [
        {
          "prompt": "[g c d e] &nbsp [a c d e]<br>[d e f q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[v w x y] &nbsp [u w x y]<br>[z o p n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[g c d e] &nbsp [a c d e]<br>[u v w x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[d e f q] &nbsp [c e f q]<br>[g c d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[g c d e] &nbsp [a c d e]<br>[i j k l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[h i j k] &nbsp [q i j k]<br>[o p n r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[h i j k] &nbsp [q i j k]<br>[e f q h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[t u v w] &nbsp [s u v w]<br>[o p n r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[h i j k] &nbsp [q i j k]<br>[d e f q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[q h i j] &nbsp [f h i j]<br>[z o p n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "add_letter": [
        {
          "prompt": "[l m z o] &nbsp [l m z o p]<br>[u v w x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[e f q h] &nbsp [e f q h i]<br>[f q h i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[g c d e] &nbsp [g c d e f]<br>[d e f q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[m z o p] &nbsp [m z o p n]<br>[e f q h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[l m z o] &nbsp [l m z o p]<br>[h i j k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[m z o p] &nbsp [m z o p n]<br>[t u v w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[n r s t] &nbsp [n r s t u]<br>[g c d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[t u v w] &nbsp [t u v w x]<br>[a g c d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[u v w x] &nbsp [u v w x y]<br>[a g c d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[p n r s] &nbsp [p n r s t]<br>[m z o p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "remove_redundant": [
        {
          "prompt": "[o p n r r s] &nbsp [o p n r s]<br>[e f q h i i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[z o p p n r] &nbsp [z o p n r]<br>[f f q h i j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[z o p n r r] &nbsp [z o p n r]<br>[p n r r s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[s s t u v w] &nbsp [s t u v w]<br>[p n r s t t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[e e f q h i] &nbsp [e f q h i]<br>[d d e f q h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[n r s t u u] &nbsp [n r s t u]<br>[z o p n n r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[c c d e f q] &nbsp [c d e f q]<br>[u v w x x y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[c d e f f q] &nbsp [c d e f q]<br>[l m m z o p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[z z o p n r] &nbsp [z o p n r]<br>[v w x y y b] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[e f q h i i] &nbsp [e f q h i]<br>[v v w x y b] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "fix_alphabet": [
        {
          "prompt": "[l m z o h] &nbsp [l m z o p]<br>[y t u v w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[x p n r s] &nbsp [o p n r s]<br>[e w x y b] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[w q h i j] &nbsp [f q h i j]<br>[q h i y k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[s t m v w] &nbsp [s t u v w]<br>[v w a y b] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[m z o s n] &nbsp [m z o p n]<br>[u f q h i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[m z o p v] &nbsp [m z o p n]<br>[c d e r q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[h n j k l] &nbsp [h i j k l]<br>[x z o p n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[l p n r s] &nbsp [o p n r s]<br>[s g c d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[e f r h i] &nbsp [e f q h i]<br>[y h i j k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[j y l m z] &nbsp [j k l m z]<br>[g c d o f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "sort": [
        {
          "prompt": "[z r p n o] &nbsp [z o p n r]<br>[i l k j m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[l i j k h] &nbsp [h i j k l]<br>[m z n p o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[n r u t s] &nbsp [n r s t u]<br>[e d f q h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[z o r n p] &nbsp [z o p n r]<br>[v w b y x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[t x v w u] &nbsp [t u v w x]<br>[a g e d c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[e g c d a] &nbsp [a g c d e]<br>[f i h q j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[l m o z p] &nbsp [l m z o p]<br>[q e f d h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[p z o m n] &nbsp [m z o p n]<br>[i h q j k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[f q i h j] &nbsp [f q h i j]<br>[s n r p t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[n s r t u] &nbsp [n r s t u]<br>[m j k l i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "attn": [
        {
          "prompt": "[g c d e] &nbsp [g c d f]<br>[t u v w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[m z o p] &nbsp [m z o n]<br>[c d e f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[e f q h] &nbsp [e f q i]<br>[g c d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[v w x y] &nbsp [v w x b]<br>[i j k l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[m z o p] &nbsp [m z o n]<br>[a g c d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[h i j k] &nbsp [h i j l]<br>[j k l m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[a g c d] &nbsp [a g c e]<br>[s t u v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[m z o p] &nbsp [m z o n]<br>[h i j k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[l m z o] &nbsp [l m z p]<br>[m z o p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[q h i j] &nbsp [q h i k]<br>[m z o p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ]
    },
    "alph_4": {
      "shuffled_letters": [
        "f",
        "k",
        "r",
        "e",
        "p"
      ],
      "shuffled_alphabet": [
        "a",
        "b",
        "c",
        "d",
        "f",
        "k",
        "g",
        "h",
        "i",
        "j",
        "r",
        "l",
        "m",
        "n",
        "o",
        "e",
        "q",
        "p",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z"
      ],
      "succ": [
        {
          "prompt": "[o e q p] &nbsp [o e q s]<br>[f k g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[l m n o] &nbsp [l m n e]<br>[f k g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[j r l m] &nbsp [j r l n]<br>[e q p s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[j r l m] &nbsp [j r l n]<br>[n o e q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[q p s t] &nbsp [q p s u]<br>[d f k g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[h i j r] &nbsp [h i j l]<br>[e q p s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[s t u v] &nbsp [s t u w]<br>[b c d f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[e q p s] &nbsp [e q p t]<br>[h i j r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[t u v w] &nbsp [t u v x]<br>[n o e q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[q p s t] &nbsp [q p s u]<br>[m n o e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "pred": [
        {
          "prompt": "[o e q p] &nbsp [n e q p]<br>[s t u v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[h i j r] &nbsp [g i j r]<br>[f k g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[k g h i] &nbsp [f g h i]<br>[m n o e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[s t u v] &nbsp [p t u v]<br>[d f k g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[u v w x] &nbsp [t v w x]<br>[f k g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[o e q p] &nbsp [n e q p]<br>[v w x y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[h i j r] &nbsp [g i j r]<br>[p s t u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[n o e q] &nbsp [m o e q]<br>[q p s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[s t u v] &nbsp [p t u v]<br>[b c d f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[s t u v] &nbsp [p t u v]<br>[p s t u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "add_letter": [
        {
          "prompt": "[r l m n] &nbsp [r l m n o]<br>[j r l m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[e q p s] &nbsp [e q p s t]<br>[u v w x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[l m n o] &nbsp [l m n o e]<br>[e q p s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[n o e q] &nbsp [n o e q p]<br>[j r l m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[e q p s] &nbsp [e q p s t]<br>[r l m n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[o e q p] &nbsp [o e q p s]<br>[a b c d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[f k g h] &nbsp [f k g h i]<br>[j r l m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[k g h i] &nbsp [k g h i j]<br>[q p s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[b c d f] &nbsp [b c d f k]<br>[l m n o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[d f k g] &nbsp [d f k g h]<br>[i j r l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "remove_redundant": [
        {
          "prompt": "[o e q q p s] &nbsp [o e q p s]<br>[d f k k g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[d f k g g h] &nbsp [d f k g h]<br>[a b b c d f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[d f f k g h] &nbsp [d f k g h]<br>[b c d f k k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[p s t t u v] &nbsp [p s t u v]<br>[o e q p s s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[u u v w x y] &nbsp [u v w x y]<br>[j r l l m n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[b c d d f k] &nbsp [b c d f k]<br>[r l m n o o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[e q p p s t] &nbsp [e q p s t]<br>[l m m n o e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[j j r l m n] &nbsp [j r l m n]<br>[h h i j r l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[d f f k g h] &nbsp [d f k g h]<br>[g h i j j r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[l l m n o e] &nbsp [l m n o e]<br>[u v v w x y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "fix_alphabet": [
        {
          "prompt": "[l m n g e] &nbsp [l m n o e]<br>[o e i p s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[b p d f k] &nbsp [b c d f k]<br>[f k g h c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[v n o e q] &nbsp [m n o e q]<br>[i j s l m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[i e q p s] &nbsp [o e q p s]<br>[n o e q g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[k g h i a] &nbsp [k g h i j]<br>[d e k g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[k g s i j] &nbsp [k g h i j]<br>[o e z p s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[x l m n o] &nbsp [r l m n o]<br>[c e f k g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[a b l d f] &nbsp [a b c d f]<br>[h w j r l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[e p s t u] &nbsp [q p s t u]<br>[k p h i j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[o e b p s] &nbsp [o e q p s]<br>[b c d f j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "sort": [
        {
          "prompt": "[w u v t x] &nbsp [t u v w x]<br>[e q t s p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[t u x w v] &nbsp [t u v w x]<br>[o n e q p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[a b c f d] &nbsp [a b c d f]<br>[s q p e t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[l m o n e] &nbsp [l m n o e]<br>[f k g i h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[c b a d f] &nbsp [a b c d f]<br>[m l r n o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[e m n o l] &nbsp [l m n o e]<br>[l i j r h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[b c k f d] &nbsp [b c d f k]<br>[d b c a f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[e q t s p] &nbsp [e q p s t]<br>[q o e n p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[k c d f b] &nbsp [b c d f k]<br>[e q p t s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[r j i l m] &nbsp [i j r l m]<br>[b d c f k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "attn": [
        {
          "prompt": "[c d f k] &nbsp [c d f g]<br>[i j r l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[h i j r] &nbsp [h i j l]<br>[e q p s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[o e q p] &nbsp [o e q s]<br>[f k g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[c d f k] &nbsp [c d f g]<br>[h i j r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[e q p s] &nbsp [e q p t]<br>[q p s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[g h i j] &nbsp [g h i r]<br>[i j r l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[c d f k] &nbsp [c d f g]<br>[p s t u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[q p s t] &nbsp [q p s u]<br>[j r l m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[c d f k] &nbsp [c d f g]<br>[e q p s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[h i j r] &nbsp [h i j l]<br>[b c d f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ]
    },
    "alph_5": {
      "shuffled_letters": [
        "k",
        "z",
        "y",
        "v",
        "i"
      ],
      "shuffled_alphabet": [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "k",
        "j",
        "z",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "y",
        "w",
        "x",
        "v",
        "i"
      ],
      "succ": [
        {
          "prompt": "[r s t u] &nbsp [r s t y]<br>[y w x v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[k j z l] &nbsp [k j z m]<br>[p q r s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[o p q r] &nbsp [o p q s]<br>[h k j z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[y w x v] &nbsp [y w x i]<br>[m n o p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[z l m n] &nbsp [z l m o]<br>[a b c d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[t u y w] &nbsp [t u y x]<br>[y w x v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[k j z l] &nbsp [k j z m]<br>[u y w x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[k j z l] &nbsp [k j z m]<br>[y w x v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[m n o p] &nbsp [m n o q]<br>[k j z l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[h k j z] &nbsp [h k j l]<br>[u y w x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "pred": [
        {
          "prompt": "[d e f g] &nbsp [c e f g]<br>[w x v i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[t u y w] &nbsp [s u y w]<br>[n o p q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[e f g h] &nbsp [d f g h]<br>[h k j z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[y w x v] &nbsp [u w x v]<br>[w x v i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[r s t u] &nbsp [q s t u]<br>[y w x v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[f g h k] &nbsp [e g h k]<br>[y w x v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[w x v i] &nbsp [y x v i]<br>[s t u y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[h k j z] &nbsp [g k j z]<br>[j z l m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[e f g h] &nbsp [d f g h]<br>[k j z l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[b c d e] &nbsp [a c d e]<br>[w x v i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "add_letter": [
        {
          "prompt": "[h k j z] &nbsp [h k j z l]<br>[o p q r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[r s t u] &nbsp [r s t u y]<br>[l m n o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[d e f g] &nbsp [d e f g h]<br>[y w x v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[j z l m] &nbsp [j z l m n]<br>[m n o p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[g h k j] &nbsp [g h k j z]<br>[o p q r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[r s t u] &nbsp [r s t u y]<br>[b c d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[y w x v] &nbsp [y w x v i]<br>[q r s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[y w x v] &nbsp [y w x v i]<br>[f g h k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[e f g h] &nbsp [e f g h k]<br>[f g h k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[y w x v] &nbsp [y w x v i]<br>[d e f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "remove_redundant": [
        {
          "prompt": "[y w w x v i] &nbsp [y w x v i]<br>[u y y w x v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[t t u y w x] &nbsp [t u y w x]<br>[e f f g h k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[y w x v v i] &nbsp [y w x v i]<br>[g h k k j z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[j z l m m n] &nbsp [j z l m n]<br>[y w x v i i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[g h h k j z] &nbsp [g h k j z]<br>[y w w x v i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[y w x v i i] &nbsp [y w x v i]<br>[k j z l l m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[n o p p q r] &nbsp [n o p q r]<br>[g h h k j z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[l l m n o p] &nbsp [l m n o p]<br>[y w x x v i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[j j z l m n] &nbsp [j z l m n]<br>[s s t u y w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[y w x v i i] &nbsp [y w x v i]<br>[c c d e f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "fix_alphabet": [
        {
          "prompt": "[h k j z e] &nbsp [h k j z l]<br>[y w h v i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[k j z i m] &nbsp [k j z l m]<br>[y w x v f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[a b c d j] &nbsp [a b c d e]<br>[n y w x v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[h k j a l] &nbsp [h k j z l]<br>[j z c m n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[y w x f i] &nbsp [y w x v i]<br>[b c d k f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[h k j z n] &nbsp [h k j z l]<br>[o p q l s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[z l m n q] &nbsp [z l m n o]<br>[k j z l p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[u l w x v] &nbsp [u y w x v]<br>[y w x v c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[c d e p g] &nbsp [c d e f g]<br>[y w g v i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[r g h k j] &nbsp [f g h k j]<br>[y w x a i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "sort": [
        {
          "prompt": "[y i x v w] &nbsp [y w x v i]<br>[j z n m l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[y v x w i] &nbsp [y w x v i]<br>[o l m n z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[u w y x v] &nbsp [u y w x v]<br>[i w x v y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[j g h k f] &nbsp [f g h k j]<br>[l z j m n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[k h j z l] &nbsp [h k j z l]<br>[l z j m n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[h k z j l] &nbsp [h k j z l]<br>[t u y x w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[c b d e f] &nbsp [b c d e f]<br>[u y x w v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[i w x v y] &nbsp [y w x v i]<br>[t x y w u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[u y x w v] &nbsp [u y w x v]<br>[h k j l z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[o n p q r] &nbsp [n o p q r]<br>[y v x w i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "attn": [
        {
          "prompt": "[u y w x] &nbsp [u y w v]<br>[s t u y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[r s t u] &nbsp [r s t y]<br>[e f g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[y w x v] &nbsp [y w x i]<br>[h k j z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[y w x v] &nbsp [y w x i]<br>[k j z l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[l m n o] &nbsp [l m n p]<br>[y w x v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[f g h k] &nbsp [f g h j]<br>[y w x v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[u y w x] &nbsp [u y w v]<br>[f g h k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[t u y w] &nbsp [t u y x]<br>[y w x v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[h k j z] &nbsp [h k j l]<br>[u y w x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[y w x v] &nbsp [y w x i]<br>[a b c d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ]
    },
    "alph_6": {
      "shuffled_letters": [
        "n",
        "r",
        "a",
        "k",
        "o"
      ],
      "shuffled_alphabet": [
        "n",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "r",
        "l",
        "m",
        "a",
        "k",
        "p",
        "q",
        "o",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z"
      ],
      "succ": [
        {
          "prompt": "[s t u v] &nbsp [s t u w]<br>[m a k p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[q o s t] &nbsp [q o s u]<br>[s t u v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[m a k p] &nbsp [m a k q]<br>[n b c d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[k p q o] &nbsp [k p q s]<br>[t u v w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[n b c d] &nbsp [n b c e]<br>[m a k p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[p q o s] &nbsp [p q o t]<br>[r l m a] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[c d e f] &nbsp [c d e g]<br>[q o s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[r l m a] &nbsp [r l m k]<br>[p q o s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[n b c d] &nbsp [n b c e]<br>[f g h i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[u v w x] &nbsp [u v w y]<br>[j r l m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "pred": [
        {
          "prompt": "[o s t u] &nbsp [q s t u]<br>[i j r l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[k p q o] &nbsp [a p q o]<br>[g h i j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[b c d e] &nbsp [n c d e]<br>[g h i j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[k p q o] &nbsp [a p q o]<br>[a k p q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[b c d e] &nbsp [n c d e]<br>[q o s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[j r l m] &nbsp [i r l m]<br>[r l m a] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[k p q o] &nbsp [a p q o]<br>[c d e f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[w x y z] &nbsp [v x y z]<br>[i j r l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[e f g h] &nbsp [d f g h]<br>[r l m a] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[i j r l] &nbsp [h j r l]<br>[d e f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "add_letter": [
        {
          "prompt": "[p q o s] &nbsp [p q o s t]<br>[v w x y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[k p q o] &nbsp [k p q o s]<br>[s t u v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[g h i j] &nbsp [g h i j r]<br>[p q o s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[h i j r] &nbsp [h i j r l]<br>[t u v w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[n b c d] &nbsp [n b c d e]<br>[a k p q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[n b c d] &nbsp [n b c d e]<br>[v w x y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[f g h i] &nbsp [f g h i j]<br>[q o s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[r l m a] &nbsp [r l m a k]<br>[g h i j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[m a k p] &nbsp [m a k p q]<br>[n b c d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[a k p q] &nbsp [a k p q o]<br>[o s t u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "remove_redundant": [
        {
          "prompt": "[m a k k p q] &nbsp [m a k p q]<br>[e e f g h i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[n b b c d e] &nbsp [n b c d e]<br>[m m a k p q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[h i j r r l] &nbsp [h i j r l]<br>[g h i j j r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[a a k p q o] &nbsp [a k p q o]<br>[c d d e f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[m a k p q q] &nbsp [m a k p q]<br>[g h i j j r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[g h i j j r] &nbsp [g h i j r]<br>[h i j r l l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[r l m a a k] &nbsp [r l m a k]<br>[o s t u u v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[o s s t u v] &nbsp [o s t u v]<br>[c d e e f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[b c c d e f] &nbsp [b c d e f]<br>[o s t u v v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[p q o s s t] &nbsp [p q o s t]<br>[t u u v w x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "fix_alphabet": [
        {
          "prompt": "[k b q o s] &nbsp [k p q o s]<br>[g l i j r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[f r l m a] &nbsp [j r l m a]<br>[a k p g o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[q o h t u] &nbsp [q o s t u]<br>[j r l i a] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[i j e l m] &nbsp [i j r l m]<br>[f s t u v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[l j a k p] &nbsp [l m a k p]<br>[m a k r q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[c p e f g] &nbsp [c d e f g]<br>[a n p q o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[o q t u v] &nbsp [o s t u v]<br>[f g l i j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[w b c d e] &nbsp [n b c d e]<br>[i j c l m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[g n i j r] &nbsp [g h i j r]<br>[m a k w q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[s t u m w] &nbsp [s t u v w]<br>[j r l m v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "sort": [
        {
          "prompt": "[o v t u s] &nbsp [o s t u v]<br>[t o s q u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[k a p q o] &nbsp [a k p q o]<br>[j i r l m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[l m k a p] &nbsp [l m a k p]<br>[l r j m a] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[r m l a k] &nbsp [r l m a k]<br>[x w v y z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[u s t o v] &nbsp [o s t u v]<br>[m a p k q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[l i j r h] &nbsp [h i j r l]<br>[e c d b f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[p s o q t] &nbsp [p q o s t]<br>[k q p o s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[e c d b f] &nbsp [b c d e f]<br>[k o q p s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[v s t u o] &nbsp [o s t u v]<br>[v x w y z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[b f d e c] &nbsp [b c d e f]<br>[n b e d c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "attn": [
        {
          "prompt": "[b c d e] &nbsp [b c d f]<br>[n b c d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[e f g h] &nbsp [e f g i]<br>[n b c d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[b c d e] &nbsp [b c d f]<br>[r l m a] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[q o s t] &nbsp [q o s u]<br>[n b c d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[p q o s] &nbsp [p q o t]<br>[r l m a] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[h i j r] &nbsp [h i j l]<br>[c d e f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[o s t u] &nbsp [o s t v]<br>[k p q o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[p q o s] &nbsp [p q o t]<br>[n b c d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[g h i j] &nbsp [g h i r]<br>[h i j r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[a k p q] &nbsp [a k p o]<br>[m a k p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ]
    }
  },
  "np_10": {
    "alph_0": {
      "shuffled_letters": [
        "q",
        "o",
        "n",
        "u",
        "f",
        "y",
        "w",
        "b",
        "z",
        "j"
      ],
      "shuffled_alphabet": [
        "a",
        "q",
        "c",
        "d",
        "e",
        "o",
        "g",
        "h",
        "i",
        "n",
        "k",
        "l",
        "m",
        "u",
        "f",
        "p",
        "y",
        "r",
        "s",
        "t",
        "w",
        "v",
        "b",
        "x",
        "z",
        "j"
      ],
      "succ": [
        {
          "prompt": "[o g h i] &nbsp [o g h n]<br>[v b x z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[n k l m] &nbsp [n k l u]<br>[v b x z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[a q c d] &nbsp [a q c e]<br>[r s t w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[k l m u] &nbsp [k l m f]<br>[q c d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[m u f p] &nbsp [m u f y]<br>[t w v b] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[f p y r] &nbsp [f p y s]<br>[l m u f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[k l m u] &nbsp [k l m f]<br>[n k l m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[g h i n] &nbsp [g h i k]<br>[a q c d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[p y r s] &nbsp [p y r t]<br>[q c d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[v b x z] &nbsp [v b x j]<br>[n k l m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "pred": [
        {
          "prompt": "[b x z j] &nbsp [v x z j]<br>[e o g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[p y r s] &nbsp [f y r s]<br>[c d e o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[b x z j] &nbsp [v x z j]<br>[v b x z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[m u f p] &nbsp [l u f p]<br>[b x z j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[q c d e] &nbsp [a c d e]<br>[y r s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[t w v b] &nbsp [s w v b]<br>[v b x z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[r s t w] &nbsp [y s t w]<br>[i n k l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[m u f p] &nbsp [l u f p]<br>[e o g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[f p y r] &nbsp [u p y r]<br>[m u f p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[y r s t] &nbsp [p r s t]<br>[s t w v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "add_letter": [
        {
          "prompt": "[u f p y] &nbsp [u f p y r]<br>[d e o g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[r s t w] &nbsp [r s t w v]<br>[g h i n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[u f p y] &nbsp [u f p y r]<br>[t w v b] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[g h i n] &nbsp [g h i n k]<br>[w v b x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[l m u f] &nbsp [l m u f p]<br>[u f p y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[v b x z] &nbsp [v b x z j]<br>[y r s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[t w v b] &nbsp [t w v b x]<br>[v b x z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[k l m u] &nbsp [k l m u f]<br>[h i n k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[g h i n] &nbsp [g h i n k]<br>[a q c d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[p y r s] &nbsp [p y r s t]<br>[w v b x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "remove_redundant": [
        {
          "prompt": "[t t w v b x] &nbsp [t w v b x]<br>[e e o g h i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[f p y r r s] &nbsp [f p y r s]<br>[w v b b x z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[l m u u f p] &nbsp [l m u f p]<br>[o g h i n n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[k l m u f f] &nbsp [k l m u f]<br>[l m m u f p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[i i n k l m] &nbsp [i n k l m]<br>[h i n n k l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[y r s t t w] &nbsp [y r s t w]<br>[p y r r s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[v b x z z j] &nbsp [v b x z j]<br>[l l m u f p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[c d d e o g] &nbsp [c d e o g]<br>[v b x z z j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[a q c d e e] &nbsp [a q c d e]<br>[n k k l m u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[v b x z z j] &nbsp [v b x z j]<br>[l m m u f p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "fix_alphabet": [
        {
          "prompt": "[k s m u f] &nbsp [k l m u f]<br>[g r i n k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[k l q u f] &nbsp [k l m u f]<br>[d v b x z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[t w j b x] &nbsp [t w v b x]<br>[z n k l m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[k r m u f] &nbsp [k l m u f]<br>[e o g a i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[p l m u f] &nbsp [k l m u f]<br>[c d e o z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[d e o s h] &nbsp [d e o g h]<br>[c d e o y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[k l m u w] &nbsp [k l m u f]<br>[u f p y l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[m u f p b] &nbsp [m u f p y]<br>[f b y r s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[l m i f p] &nbsp [l m u f p]<br>[i n k w m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[k g m u f] &nbsp [k l m u f]<br>[a f p y r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "sort": [
        {
          "prompt": "[e c d q o] &nbsp [q c d e o]<br>[j b x z v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[g e o d h] &nbsp [d e o g h]<br>[v x b z j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[a c q d e] &nbsp [a q c d e]<br>[l m p f u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[f u m p y] &nbsp [m u f p y]<br>[i n l k m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[o c d e q] &nbsp [q c d e o]<br>[h e o g d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[i l k n m] &nbsp [i n k l m]<br>[b w v t x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[f p s r y] &nbsp [f p y r s]<br>[y t s r w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[q c o e d] &nbsp [q c d e o]<br>[a q d c e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[u r p y f] &nbsp [u f p y r]<br>[t w v x b] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[f p r y s] &nbsp [f p y r s]<br>[d o e g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "attn": [
        {
          "prompt": "[v b x z] &nbsp [v b x j]<br>[r s t w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[u f p y] &nbsp [u f p r]<br>[t w v b] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[g h i n] &nbsp [g h i k]<br>[l m u f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[t w v b] &nbsp [t w v x]<br>[g h i n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[a q c d] &nbsp [a q c e]<br>[p y r s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[f p y r] &nbsp [f p y s]<br>[q c d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[m u f p] &nbsp [m u f y]<br>[a q c d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[a q c d] &nbsp [a q c e]<br>[w v b x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[s t w v] &nbsp [s t w b]<br>[l m u f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[g h i n] &nbsp [g h i k]<br>[y r s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ]
    },
    "alph_1": {
      "shuffled_letters": [
        "q",
        "y",
        "o",
        "t",
        "x",
        "h",
        "u",
        "d",
        "m",
        "j"
      ],
      "shuffled_alphabet": [
        "a",
        "b",
        "c",
        "q",
        "e",
        "f",
        "g",
        "y",
        "i",
        "o",
        "k",
        "l",
        "t",
        "n",
        "x",
        "p",
        "h",
        "r",
        "s",
        "u",
        "d",
        "v",
        "w",
        "m",
        "j",
        "z"
      ],
      "succ": [
        {
          "prompt": "[p h r s] &nbsp [p h r u]<br>[r s u d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[i o k l] &nbsp [i o k t]<br>[s u d v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[r s u d] &nbsp [r s u v]<br>[p h r s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[k l t n] &nbsp [k l t x]<br>[q e f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[x p h r] &nbsp [x p h s]<br>[g y i o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[v w m j] &nbsp [v w m z]<br>[s u d v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[h r s u] &nbsp [h r s d]<br>[t n x p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[f g y i] &nbsp [f g y o]<br>[b c q e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[v w m j] &nbsp [v w m z]<br>[c q e f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[u d v w] &nbsp [u d v m]<br>[h r s u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "pred": [
        {
          "prompt": "[y i o k] &nbsp [g i o k]<br>[p h r s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[i o k l] &nbsp [y o k l]<br>[h r s u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[u d v w] &nbsp [s d v w]<br>[i o k l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[f g y i] &nbsp [e g y i]<br>[r s u d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[o k l t] &nbsp [i k l t]<br>[y i o k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[s u d v] &nbsp [r u d v]<br>[u d v w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[x p h r] &nbsp [n p h r]<br>[l t n x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[r s u d] &nbsp [h s u d]<br>[e f g y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[n x p h] &nbsp [t x p h]<br>[e f g y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[w m j z] &nbsp [v m j z]<br>[c q e f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "add_letter": [
        {
          "prompt": "[i o k l] &nbsp [i o k l t]<br>[x p h r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[r s u d] &nbsp [r s u d v]<br>[y i o k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[h r s u] &nbsp [h r s u d]<br>[b c q e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[v w m j] &nbsp [v w m j z]<br>[n x p h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[n x p h] &nbsp [n x p h r]<br>[v w m j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[e f g y] &nbsp [e f g y i]<br>[p h r s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[a b c q] &nbsp [a b c q e]<br>[h r s u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[e f g y] &nbsp [e f g y i]<br>[r s u d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[s u d v] &nbsp [s u d v w]<br>[v w m j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[k l t n] &nbsp [k l t n x]<br>[u d v w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "remove_redundant": [
        {
          "prompt": "[v w m j z z] &nbsp [v w m j z]<br>[p h r s u u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[g g y i o k] &nbsp [g y i o k]<br>[h r r s u d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[n x p h h r] &nbsp [n x p h r]<br>[f g y i o o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[h r s u u d] &nbsp [h r s u d]<br>[k l t t n x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[l t n x p p] &nbsp [l t n x p]<br>[f g g y i o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[x p h r r s] &nbsp [x p h r s]<br>[l t t n x p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[r s u d v v] &nbsp [r s u d v]<br>[v w m j z z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[g y i o k k] &nbsp [g y i o k]<br>[r s u u d v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[o o k l t n] &nbsp [o k l t n]<br>[v w m j z z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[c c q e f g] &nbsp [c q e f g]<br>[o k l t n n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "fix_alphabet": [
        {
          "prompt": "[g n i o k] &nbsp [g y i o k]<br>[m l t n x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[h b c q e] &nbsp [a b c q e]<br>[b l t n x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[g u d v w] &nbsp [s u d v w]<br>[h r s u q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[x p i r s] &nbsp [x p h r s]<br>[t n x p o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[y i h k l] &nbsp [y i o k l]<br>[e f p y i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[e f n y i] &nbsp [e f g y i]<br>[p b r s u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[t n x p q] &nbsp [t n x p h]<br>[u d y w m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[f r y i o] &nbsp [f g y i o]<br>[t n x p c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[h r s u e] &nbsp [h r s u d]<br>[k l t f x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[r s u d i] &nbsp [r s u d v]<br>[k r s u d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "sort": [
        {
          "prompt": "[h p x r s] &nbsp [x p h r s]<br>[g f y i o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[v j m w z] &nbsp [v w m j z]<br>[a b c e q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[w u d v s] &nbsp [s u d v w]<br>[y g f i o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[k o l t n] &nbsp [o k l t n]<br>[c q e g f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[t l n x p] &nbsp [l t n x p]<br>[l o k i t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[n k l t o] &nbsp [o k l t n]<br>[p n x t h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[u v d w m] &nbsp [u d v w m]<br>[k o l t n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[d r s u h] &nbsp [h r s u d]<br>[o k t l n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[v m w j z] &nbsp [v w m j z]<br>[x p h s r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[g f e y i] &nbsp [e f g y i]<br>[r s d u v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "attn": [
        {
          "prompt": "[b c q e] &nbsp [b c q f]<br>[t n x p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[e f g y] &nbsp [e f g i]<br>[h r s u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[v w m j] &nbsp [v w m z]<br>[o k l t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[t n x p] &nbsp [t n x h]<br>[c q e f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[v w m j] &nbsp [v w m z]<br>[d v w m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[c q e f] &nbsp [c q e g]<br>[v w m j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[y i o k] &nbsp [y i o l]<br>[d v w m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[u d v w] &nbsp [u d v m]<br>[a b c q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[s u d v] &nbsp [s u d w]<br>[x p h r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[u d v w] &nbsp [u d v m]<br>[t n x p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ]
    },
    "alph_2": {
      "shuffled_letters": [
        "c",
        "v",
        "h",
        "d",
        "r",
        "y",
        "x",
        "a",
        "w",
        "i"
      ],
      "shuffled_alphabet": [
        "c",
        "b",
        "v",
        "h",
        "e",
        "f",
        "g",
        "d",
        "r",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "y",
        "s",
        "t",
        "u",
        "x",
        "a",
        "w",
        "i",
        "z"
      ],
      "succ": [
        {
          "prompt": "[f g d r] &nbsp [f g d j]<br>[b v h e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[p q y s] &nbsp [p q y t]<br>[h e f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[c b v h] &nbsp [c b v e]<br>[r j k l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[q y s t] &nbsp [q y s u]<br>[f g d r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[n o p q] &nbsp [n o p y]<br>[b v h e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[c b v h] &nbsp [c b v e]<br>[q y s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[m n o p] &nbsp [m n o q]<br>[u x a w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[b v h e] &nbsp [b v h f]<br>[x a w i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[s t u x] &nbsp [s t u a]<br>[l m n o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[c b v h] &nbsp [c b v e]<br>[o p q y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "pred": [
        {
          "prompt": "[u x a w] &nbsp [t x a w]<br>[d r j k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[e f g d] &nbsp [h f g d]<br>[f g d r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[v h e f] &nbsp [b h e f]<br>[e f g d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[j k l m] &nbsp [r k l m]<br>[u x a w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[n o p q] &nbsp [m o p q]<br>[j k l m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[l m n o] &nbsp [k m n o]<br>[v h e f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[u x a w] &nbsp [t x a w]<br>[r j k l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[y s t u] &nbsp [q s t u]<br>[n o p q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[x a w i] &nbsp [u a w i]<br>[p q y s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[b v h e] &nbsp [c v h e]<br>[y s t u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "add_letter": [
        {
          "prompt": "[s t u x] &nbsp [s t u x a]<br>[u x a w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[x a w i] &nbsp [x a w i z]<br>[n o p q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[e f g d] &nbsp [e f g d r]<br>[c b v h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[o p q y] &nbsp [o p q y s]<br>[s t u x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[j k l m] &nbsp [j k l m n]<br>[x a w i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[t u x a] &nbsp [t u x a w]<br>[m n o p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[x a w i] &nbsp [x a w i z]<br>[m n o p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[m n o p] &nbsp [m n o p q]<br>[b v h e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[q y s t] &nbsp [q y s t u]<br>[m n o p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[y s t u] &nbsp [y s t u x]<br>[u x a w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "remove_redundant": [
        {
          "prompt": "[s t t u x a] &nbsp [s t u x a]<br>[x a w i i z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[v v h e f g] &nbsp [v h e f g]<br>[p q y s t t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[v h e f g g] &nbsp [v h e f g]<br>[t t u x a w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[e f g g d r] &nbsp [e f g d r]<br>[x x a w i z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[o p q q y s] &nbsp [o p q y s]<br>[n o o p q y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[b v h e e f] &nbsp [b v h e f]<br>[p q y s t t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[t t u x a w] &nbsp [t u x a w]<br>[k l m m n o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[x x a w i z] &nbsp [x a w i z]<br>[v h h e f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[x a w i i z] &nbsp [x a w i z]<br>[k k l m n o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[e e f g d r] &nbsp [e f g d r]<br>[s t u u x a] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "fix_alphabet": [
        {
          "prompt": "[k s m n o] &nbsp [k l m n o]<br>[x c w i z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[b a h e f] &nbsp [b v h e f]<br>[y s v u x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[f m d r j] &nbsp [f g d r j]<br>[y s p u x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[c b v o e] &nbsp [c b v h e]<br>[s t u x g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[s z u x a] &nbsp [s t u x a]<br>[y s z u x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[c y s t u] &nbsp [q y s t u]<br>[u r a w i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[x h w i z] &nbsp [x a w i z]<br>[v h e f b] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[x o w i z] &nbsp [x a w i z]<br>[p b y s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[c b v w e] &nbsp [c b v h e]<br>[b o h e f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[s t u x c] &nbsp [s t u x a]<br>[q s t u x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "sort": [
        {
          "prompt": "[v e h f g] &nbsp [v h e f g]<br>[q s y t u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[b e h v f] &nbsp [b v h e f]<br>[d g f r j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[h g f e d] &nbsp [h e f g d]<br>[a x w i z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[d f g e r] &nbsp [e f g d r]<br>[t q y s p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[s a u x t] &nbsp [s t u x a]<br>[y p q o s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[o y q p s] &nbsp [o p q y s]<br>[c b v e h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[j m l k n] &nbsp [j k l m n]<br>[x a z i w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[b v h f e] &nbsp [b v h e f]<br>[d g f r j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[t w x a u] &nbsp [t u x a w]<br>[p o n q y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[n q p o y] &nbsp [n o p q y]<br>[e r g d f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "attn": [
        {
          "prompt": "[p q y s] &nbsp [p q y t]<br>[x a w i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[b v h e] &nbsp [b v h f]<br>[f g d r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[v h e f] &nbsp [v h e g]<br>[e f g d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[c b v h] &nbsp [c b v e]<br>[q y s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[j k l m] &nbsp [j k l n]<br>[r j k l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[c b v h] &nbsp [c b v e]<br>[h e f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[b v h e] &nbsp [b v h f]<br>[m n o p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[f g d r] &nbsp [f g d j]<br>[h e f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[s t u x] &nbsp [s t u a]<br>[q y s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[x a w i] &nbsp [x a w z]<br>[q y s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ]
    },
    "alph_3": {
      "shuffled_letters": [
        "m",
        "p",
        "z",
        "b",
        "d",
        "k",
        "j",
        "l",
        "e",
        "y"
      ],
      "shuffled_alphabet": [
        "a",
        "m",
        "c",
        "p",
        "z",
        "f",
        "g",
        "h",
        "i",
        "b",
        "d",
        "k",
        "j",
        "n",
        "o",
        "l",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "e",
        "y"
      ],
      "succ": [
        {
          "prompt": "[z f g h] &nbsp [z f g i]<br>[q r s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[i b d k] &nbsp [i b d j]<br>[u v w x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[h i b d] &nbsp [h i b k]<br>[v w x e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[b d k j] &nbsp [b d k n]<br>[t u v w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[s t u v] &nbsp [s t u w]<br>[a m c p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[u v w x] &nbsp [u v w e]<br>[b d k j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[p z f g] &nbsp [p z f h]<br>[o l q r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[n o l q] &nbsp [n o l r]<br>[h i b d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[i b d k] &nbsp [i b d j]<br>[b d k j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[v w x e] &nbsp [v w x y]<br>[p z f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "pred": [
        {
          "prompt": "[m c p z] &nbsp [a c p z]<br>[o l q r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[i b d k] &nbsp [h b d k]<br>[r s t u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[p z f g] &nbsp [c z f g]<br>[m c p z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[d k j n] &nbsp [b k j n]<br>[k j n o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[n o l q] &nbsp [j o l q]<br>[d k j n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[k j n o] &nbsp [d j n o]<br>[w x e y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[c p z f] &nbsp [m p z f]<br>[p z f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[p z f g] &nbsp [c z f g]<br>[h i b d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[m c p z] &nbsp [a c p z]<br>[p z f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[v w x e] &nbsp [u w x e]<br>[z f g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "add_letter": [
        {
          "prompt": "[a m c p] &nbsp [a m c p z]<br>[t u v w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[c p z f] &nbsp [c p z f g]<br>[a m c p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[h i b d] &nbsp [h i b d k]<br>[c p z f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[f g h i] &nbsp [f g h i b]<br>[v w x e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[d k j n] &nbsp [d k j n o]<br>[g h i b] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[h i b d] &nbsp [h i b d k]<br>[s t u v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[v w x e] &nbsp [v w x e y]<br>[a m c p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[k j n o] &nbsp [k j n o l]<br>[f g h i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[v w x e] &nbsp [v w x e y]<br>[j n o l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[c p z f] &nbsp [c p z f g]<br>[u v w x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "remove_redundant": [
        {
          "prompt": "[z f g h h i] &nbsp [z f g h i]<br>[c p z z f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[m c p z z f] &nbsp [m c p z f]<br>[l q r r s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[a m c p p z] &nbsp [a m c p z]<br>[q r s t u u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[q q r s t u] &nbsp [q r s t u]<br>[d k j n o o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[f g h i b b] &nbsp [f g h i b]<br>[b d k j n n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[a m c p z z] &nbsp [a m c p z]<br>[v w x e y y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[j n o o l q] &nbsp [j n o l q]<br>[k j n o o l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[m m c p z f] &nbsp [m c p z f]<br>[p p z f g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[q r s t u u] &nbsp [q r s t u]<br>[g h i b b d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[h i b b d k] &nbsp [h i b d k]<br>[k j n n o l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "fix_alphabet": [
        {
          "prompt": "[b f k j n] &nbsp [b d k j n]<br>[n y l q r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[a q c p z] &nbsp [a m c p z]<br>[j y o l q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[k j n o p] &nbsp [k j n o l]<br>[v w x o y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[h i b d j] &nbsp [h i b d k]<br>[i b d t j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[s t j v w] &nbsp [s t u v w]<br>[n o a q r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[k y n o l] &nbsp [k j n o l]<br>[c l q r s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[q p z f g] &nbsp [c p z f g]<br>[j n o r q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[h i b d n] &nbsp [h i b d k]<br>[d t j n o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[h k j n o] &nbsp [d k j n o]<br>[e m c p z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[a m c p t] &nbsp [a m c p z]<br>[m i b d k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "sort": [
        {
          "prompt": "[v e x w y] &nbsp [v w x e y]<br>[b i d k j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[i g h f b] &nbsp [f g h i b]<br>[n k j d o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[v t u s w] &nbsp [s t u v w]<br>[k j l o n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[a m c z p] &nbsp [a m c p z]<br>[s q r l t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[b d j k n] &nbsp [b d k j n]<br>[k i b d h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[h z f g p] &nbsp [p z f g h]<br>[a m z p c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[a p c m z] &nbsp [a m c p z]<br>[t q r s l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[l t r s q] &nbsp [l q r s t]<br>[a z c p m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[a z c p m] &nbsp [a m c p z]<br>[t q r s l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[l o q r s] &nbsp [o l q r s]<br>[m c z p f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "attn": [
        {
          "prompt": "[n o l q] &nbsp [n o l r]<br>[j n o l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[f g h i] &nbsp [f g h b]<br>[k j n o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[a m c p] &nbsp [a m c z]<br>[b d k j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[f g h i] &nbsp [f g h b]<br>[o l q r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[h i b d] &nbsp [h i b k]<br>[g h i b] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[d k j n] &nbsp [d k j o]<br>[b d k j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[u v w x] &nbsp [u v w e]<br>[n o l q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[b d k j] &nbsp [b d k n]<br>[a m c p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[n o l q] &nbsp [n o l r]<br>[f g h i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[t u v w] &nbsp [t u v x]<br>[m c p z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ]
    },
    "alph_4": {
      "shuffled_letters": [
        "g",
        "a",
        "q",
        "t",
        "z",
        "l",
        "h",
        "n",
        "j",
        "r"
      ],
      "shuffled_alphabet": [
        "g",
        "b",
        "c",
        "d",
        "e",
        "f",
        "a",
        "q",
        "i",
        "t",
        "k",
        "z",
        "m",
        "l",
        "o",
        "p",
        "h",
        "n",
        "s",
        "j",
        "u",
        "v",
        "w",
        "x",
        "y",
        "r"
      ],
      "succ": [
        {
          "prompt": "[l o p h] &nbsp [l o p n]<br>[z m l o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[z m l o] &nbsp [z m l p]<br>[k z m l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[b c d e] &nbsp [b c d f]<br>[t k z m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[l o p h] &nbsp [l o p n]<br>[d e f a] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[p h n s] &nbsp [p h n j]<br>[h n s j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[p h n s] &nbsp [p h n j]<br>[n s j u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[u v w x] &nbsp [u v w y]<br>[p h n s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[t k z m] &nbsp [t k z l]<br>[f a q i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[m l o p] &nbsp [m l o h]<br>[s j u v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[t k z m] &nbsp [t k z l]<br>[g b c d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "pred": [
        {
          "prompt": "[o p h n] &nbsp [l p h n]<br>[p h n s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[h n s j] &nbsp [p n s j]<br>[b c d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[n s j u] &nbsp [h s j u]<br>[f a q i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[j u v w] &nbsp [s u v w]<br>[b c d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[h n s j] &nbsp [p n s j]<br>[f a q i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[u v w x] &nbsp [j v w x]<br>[b c d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[h n s j] &nbsp [p n s j]<br>[t k z m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[s j u v] &nbsp [n j u v]<br>[d e f a] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[f a q i] &nbsp [e a q i]<br>[j u v w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[s j u v] &nbsp [n j u v]<br>[b c d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "add_letter": [
        {
          "prompt": "[j u v w] &nbsp [j u v w x]<br>[e f a q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[c d e f] &nbsp [c d e f a]<br>[s j u v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[k z m l] &nbsp [k z m l o]<br>[b c d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[p h n s] &nbsp [p h n s j]<br>[t k z m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[l o p h] &nbsp [l o p h n]<br>[j u v w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[n s j u] &nbsp [n s j u v]<br>[k z m l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[n s j u] &nbsp [n s j u v]<br>[h n s j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[v w x y] &nbsp [v w x y r]<br>[a q i t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[h n s j] &nbsp [h n s j u]<br>[o p h n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[c d e f] &nbsp [c d e f a]<br>[i t k z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "remove_redundant": [
        {
          "prompt": "[k z m m l o] &nbsp [k z m l o]<br>[i t k z z m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[a a q i t k] &nbsp [a q i t k]<br>[h h n s j u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[s j u v w w] &nbsp [s j u v w]<br>[d d e f a q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[z m l l o p] &nbsp [z m l o p]<br>[v w x x y r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[c d d e f a] &nbsp [c d e f a]<br>[j u u v w x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[t k z m m l] &nbsp [t k z m l]<br>[g b c c d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[d d e f a q] &nbsp [d e f a q]<br>[m m l o p h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[a q i t k k] &nbsp [a q i t k]<br>[t k z z m l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[h n s j j u] &nbsp [h n s j u]<br>[s j u u v w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[g b c d d e] &nbsp [g b c d e]<br>[l l o p h n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "fix_alphabet": [
        {
          "prompt": "[i n k z m] &nbsp [i t k z m]<br>[v w k y r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[z m i o p] &nbsp [z m l o p]<br>[e f a c i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[a q i t j] &nbsp [a q i t k]<br>[h u v w x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[s j u v m] &nbsp [s j u v w]<br>[l b c d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[l a q i t] &nbsp [f a q i t]<br>[p h n s g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[z m s o p] &nbsp [z m l o p]<br>[i n s j u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[s j u e w] &nbsp [s j u v w]<br>[o p a n s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[a q i t x] &nbsp [a q i t k]<br>[z m d o p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[e f s q i] &nbsp [e f a q i]<br>[h n s e u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[p i n s j] &nbsp [p h n s j]<br>[u v w i y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "sort": [
        {
          "prompt": "[w j u v s] &nbsp [s j u v w]<br>[l o h p n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[c d a f e] &nbsp [c d e f a]<br>[f c d e b] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[d a f e q] &nbsp [d e f a q]<br>[o s h n p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[z l m o p] &nbsp [z m l o p]<br>[m l p o h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[s v u j w] &nbsp [s j u v w]<br>[o s h n p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[v w r y x] &nbsp [v w x y r]<br>[w u v j x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[f a q t i] &nbsp [f a q i t]<br>[o l m p h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[z m p o l] &nbsp [z m l o p]<br>[m l h p o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[j n s h u] &nbsp [h n s j u]<br>[a e f d q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[d e f q a] &nbsp [d e f a q]<br>[a f e q i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "attn": [
        {
          "prompt": "[h n s j] &nbsp [h n s u]<br>[p h n s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[q i t k] &nbsp [q i t z]<br>[h n s j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[n s j u] &nbsp [n s j v]<br>[i t k z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[q i t k] &nbsp [q i t z]<br>[n s j u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[o p h n] &nbsp [o p h s]<br>[h n s j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[v w x y] &nbsp [v w x r]<br>[u v w x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[c d e f] &nbsp [c d e a]<br>[d e f a] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[q i t k] &nbsp [q i t z]<br>[a q i t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[f a q i] &nbsp [f a q t]<br>[c d e f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[g b c d] &nbsp [g b c e]<br>[q i t k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ]
    },
    "alph_5": {
      "shuffled_letters": [
        "e",
        "a",
        "b",
        "r",
        "w",
        "p",
        "n",
        "h",
        "l",
        "o"
      ],
      "shuffled_alphabet": [
        "e",
        "a",
        "c",
        "d",
        "b",
        "f",
        "g",
        "r",
        "i",
        "j",
        "k",
        "w",
        "m",
        "p",
        "n",
        "h",
        "q",
        "l",
        "s",
        "t",
        "u",
        "v",
        "o",
        "x",
        "y",
        "z"
      ],
      "succ": [
        {
          "prompt": "[r i j k] &nbsp [r i j w]<br>[e a c d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[h q l s] &nbsp [h q l t]<br>[i j k w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[r i j k] &nbsp [r i j w]<br>[l s t u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[h q l s] &nbsp [h q l t]<br>[a c d b] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[h q l s] &nbsp [h q l t]<br>[n h q l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[k w m p] &nbsp [k w m n]<br>[p n h q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[u v o x] &nbsp [u v o y]<br>[j k w m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[e a c d] &nbsp [e a c b]<br>[u v o x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[e a c d] &nbsp [e a c b]<br>[d b f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[c d b f] &nbsp [c d b g]<br>[j k w m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "pred": [
        {
          "prompt": "[c d b f] &nbsp [a d b f]<br>[p n h q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[b f g r] &nbsp [d f g r]<br>[q l s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[v o x y] &nbsp [u o x y]<br>[p n h q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[h q l s] &nbsp [n q l s]<br>[k w m p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[t u v o] &nbsp [s u v o]<br>[c d b f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[a c d b] &nbsp [e c d b]<br>[q l s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[o x y z] &nbsp [v x y z]<br>[a c d b] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[b f g r] &nbsp [d f g r]<br>[p n h q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[f g r i] &nbsp [b g r i]<br>[w m p n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[b f g r] &nbsp [d f g r]<br>[m p n h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "add_letter": [
        {
          "prompt": "[w m p n] &nbsp [w m p n h]<br>[m p n h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[t u v o] &nbsp [t u v o x]<br>[f g r i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[t u v o] &nbsp [t u v o x]<br>[e a c d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[m p n h] &nbsp [m p n h q]<br>[t u v o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[r i j k] &nbsp [r i j k w]<br>[s t u v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[k w m p] &nbsp [k w m p n]<br>[p n h q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[e a c d] &nbsp [e a c d b]<br>[g r i j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[e a c d] &nbsp [e a c d b]<br>[p n h q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[s t u v] &nbsp [s t u v o]<br>[b f g r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[q l s t] &nbsp [q l s t u]<br>[f g r i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "remove_redundant": [
        {
          "prompt": "[a c c d b f] &nbsp [a c d b f]<br>[k w m p p n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[q l s t t u] &nbsp [q l s t u]<br>[v o o x y z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[w m p p n h] &nbsp [w m p n h]<br>[e a c d b b] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[s t u v o o] &nbsp [s t u v o]<br>[a c d d b f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[q l l s t u] &nbsp [q l s t u]<br>[e a a c d b] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[e a c c d b] &nbsp [e a c d b]<br>[a c d b b f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[s t u v v o] &nbsp [s t u v o]<br>[f g r i i j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[r r i j k w] &nbsp [r i j k w]<br>[i j k k w m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[j j k w m p] &nbsp [j k w m p]<br>[a a c d b f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[b f g r i i] &nbsp [b f g r i]<br>[c c d b f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "fix_alphabet": [
        {
          "prompt": "[r i j k b] &nbsp [r i j k w]<br>[k w r p n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[e a c z b] &nbsp [e a c d b]<br>[v o x y f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[e a c d m] &nbsp [e a c d b]<br>[i j k e m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[h d b f g] &nbsp [c d b f g]<br>[w m p c h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[v c d b f] &nbsp [a c d b f]<br>[e a h d b] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[e n c d b] &nbsp [e a c d b]<br>[g i j k w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[n h m l s] &nbsp [n h q l s]<br>[m p k h q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[h q l s n] &nbsp [h q l s t]<br>[k w m p u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[p n h q f] &nbsp [p n h q l]<br>[h j k w m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[x l s t u] &nbsp [q l s t u]<br>[a c d o f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "sort": [
        {
          "prompt": "[c f b d g] &nbsp [c d b f g]<br>[b a c d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[u s t l v] &nbsp [l s t u v]<br>[w m h n p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[b a c d e] &nbsp [e a c d b]<br>[a b d c f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[p m n h q] &nbsp [m p n h q]<br>[e b c d a] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[q l t s u] &nbsp [q l s t u]<br>[p l h q n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[q l t s u] &nbsp [q l s t u]<br>[m p q h n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[p l h q n] &nbsp [p n h q l]<br>[w k m p n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[r g i j k] &nbsp [g r i j k]<br>[h s l q t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[n p m h q] &nbsp [m p n h q]<br>[c a d b f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[f i r g j] &nbsp [f g r i j]<br>[j k m w p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "attn": [
        {
          "prompt": "[a c d b] &nbsp [a c d f]<br>[i j k w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[l s t u] &nbsp [l s t v]<br>[p n h q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[a c d b] &nbsp [a c d f]<br>[h q l s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[p n h q] &nbsp [p n h l]<br>[d b f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[p n h q] &nbsp [p n h l]<br>[u v o x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[k w m p] &nbsp [k w m n]<br>[e a c d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[l s t u] &nbsp [l s t v]<br>[j k w m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[h q l s] &nbsp [h q l t]<br>[r i j k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[e a c d] &nbsp [e a c b]<br>[w m p n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[d b f g] &nbsp [d b f r]<br>[m p n h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ]
    },
    "alph_6": {
      "shuffled_letters": [
        "s",
        "o",
        "c",
        "x",
        "h",
        "t",
        "y",
        "r",
        "l",
        "m"
      ],
      "shuffled_alphabet": [
        "a",
        "b",
        "s",
        "d",
        "e",
        "f",
        "g",
        "o",
        "i",
        "j",
        "k",
        "c",
        "x",
        "n",
        "h",
        "p",
        "q",
        "t",
        "y",
        "r",
        "u",
        "v",
        "w",
        "l",
        "m",
        "z"
      ],
      "succ": [
        {
          "prompt": "[t y r u] &nbsp [t y r v]<br>[f g o i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[i j k c] &nbsp [i j k x]<br>[o i j k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[b s d e] &nbsp [b s d f]<br>[f g o i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[o i j k] &nbsp [o i j c]<br>[v w l m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[x n h p] &nbsp [x n h q]<br>[h p q t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[s d e f] &nbsp [s d e g]<br>[h p q t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[g o i j] &nbsp [g o i k]<br>[f g o i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[c x n h] &nbsp [c x n p]<br>[u v w l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[c x n h] &nbsp [c x n p]<br>[y r u v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[u v w l] &nbsp [u v w m]<br>[g o i j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "pred": [
        {
          "prompt": "[t y r u] &nbsp [q y r u]<br>[w l m z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[c x n h] &nbsp [k x n h]<br>[g o i j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[v w l m] &nbsp [u w l m]<br>[g o i j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[p q t y] &nbsp [h q t y]<br>[e f g o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[q t y r] &nbsp [p t y r]<br>[o i j k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[x n h p] &nbsp [c n h p]<br>[e f g o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[v w l m] &nbsp [u w l m]<br>[o i j k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[f g o i] &nbsp [e g o i]<br>[v w l m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[b s d e] &nbsp [a s d e]<br>[r u v w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[h p q t] &nbsp [n p q t]<br>[r u v w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "add_letter": [
        {
          "prompt": "[h p q t] &nbsp [h p q t y]<br>[q t y r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[g o i j] &nbsp [g o i j k]<br>[r u v w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[h p q t] &nbsp [h p q t y]<br>[a b s d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[y r u v] &nbsp [y r u v w]<br>[e f g o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[g o i j] &nbsp [g o i j k]<br>[d e f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[a b s d] &nbsp [a b s d e]<br>[r u v w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[a b s d] &nbsp [a b s d e]<br>[t y r u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[j k c x] &nbsp [j k c x n]<br>[b s d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[r u v w] &nbsp [r u v w l]<br>[t y r u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[r u v w] &nbsp [r u v w l]<br>[u v w l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "remove_redundant": [
        {
          "prompt": "[g o o i j k] &nbsp [g o i j k]<br>[d e f g g o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[q t y y r u] &nbsp [q t y r u]<br>[k c x n h h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[t y r u v v] &nbsp [t y r u v]<br>[f g o o i j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[k c x n n h] &nbsp [k c x n h]<br>[a a b s d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[h p p q t y] &nbsp [h p q t y]<br>[s s d e f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[c x n n h p] &nbsp [c x n h p]<br>[g o i j k k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[h p q t t y] &nbsp [h p q t y]<br>[x n h p q q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[e f g o i i] &nbsp [e f g o i]<br>[p q t y r r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[e f g g o i] &nbsp [e f g o i]<br>[q t y r r u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[r r u v w l] &nbsp [r u v w l]<br>[h p q q t y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "fix_alphabet": [
        {
          "prompt": "[c x n e p] &nbsp [c x n h p]<br>[v b l m z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[g o i j q] &nbsp [g o i j k]<br>[k c d n h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[u v w q m] &nbsp [u v w l m]<br>[v w l m x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[b m d e f] &nbsp [b s d e f]<br>[p q v y r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[p q t y i] &nbsp [p q t y r]<br>[o h j k c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[m c x n h] &nbsp [k c x n h]<br>[e k g o i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[h n q t y] &nbsp [h p q t y]<br>[g o r j k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[k c l n h] &nbsp [k c x n h]<br>[i j z c x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[u a w l m] &nbsp [u v w l m]<br>[n h p w t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[y r u x w] &nbsp [y r u v w]<br>[o i j k v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "sort": [
        {
          "prompt": "[t v r u y] &nbsp [t y r u v]<br>[m w l v z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[h n p q t] &nbsp [n h p q t]<br>[g f o i j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[p q y t r] &nbsp [p q t y r]<br>[p h n q t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[r v u w l] &nbsp [r u v w l]<br>[h n x p q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[v y r u t] &nbsp [t y r u v]<br>[f g i o j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[i f g o e] &nbsp [e f g o i]<br>[e s d b f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[r q t y p] &nbsp [p q t y r]<br>[u v m l w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[m w l v z] &nbsp [v w l m z]<br>[p y t q r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[y p q t h] &nbsp [h p q t y]<br>[v l w m z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[b f d e s] &nbsp [b s d e f]<br>[a b e d s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "attn": [
        {
          "prompt": "[b s d e] &nbsp [b s d f]<br>[v w l m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[e f g o] &nbsp [e f g i]<br>[p q t y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[k c x n] &nbsp [k c x h]<br>[d e f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[c x n h] &nbsp [c x n p]<br>[q t y r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[d e f g] &nbsp [d e f o]<br>[s d e f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[x n h p] &nbsp [x n h q]<br>[k c x n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[b s d e] &nbsp [b s d f]<br>[i j k c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[v w l m] &nbsp [v w l z]<br>[s d e f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[t y r u] &nbsp [t y r v]<br>[q t y r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[g o i j] &nbsp [g o i k]<br>[v w l m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ]
    }
  },
  "np_20": {
    "alph_0": {
      "shuffled_letters": [
        "u",
        "n",
        "o",
        "y",
        "q",
        "l",
        "f",
        "j",
        "d",
        "h",
        "a",
        "z",
        "b",
        "e",
        "w",
        "m",
        "c",
        "i",
        "x",
        "v"
      ],
      "shuffled_alphabet": [
        "u",
        "n",
        "o",
        "y",
        "q",
        "l",
        "g",
        "f",
        "j",
        "d",
        "k",
        "h",
        "a",
        "z",
        "b",
        "p",
        "e",
        "r",
        "s",
        "t",
        "w",
        "m",
        "c",
        "i",
        "x",
        "v"
      ],
      "succ": [
        {
          "prompt": "[l g f j] &nbsp [l g f d]<br>[z b p e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[u n o y] &nbsp [u n o q]<br>[g f j d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[m c i x] &nbsp [m c i v]<br>[k h a z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[e r s t] &nbsp [e r s w]<br>[s t w m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[f j d k] &nbsp [f j d h]<br>[u n o y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[o y q l] &nbsp [o y q g]<br>[t w m c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[b p e r] &nbsp [b p e s]<br>[n o y q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[k h a z] &nbsp [k h a b]<br>[o y q l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[g f j d] &nbsp [g f j k]<br>[r s t w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[o y q l] &nbsp [o y q g]<br>[h a z b] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "pred": [
        {
          "prompt": "[a z b p] &nbsp [h z b p]<br>[d k h a] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[p e r s] &nbsp [b e r s]<br>[a z b p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[n o y q] &nbsp [u o y q]<br>[c i x v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[c i x v] &nbsp [m i x v]<br>[z b p e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[r s t w] &nbsp [e s t w]<br>[g f j d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[m c i x] &nbsp [w c i x]<br>[e r s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[t w m c] &nbsp [s w m c]<br>[e r s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[e r s t] &nbsp [p r s t]<br>[z b p e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[c i x v] &nbsp [m i x v]<br>[g f j d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[z b p e] &nbsp [a b p e]<br>[o y q l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "add_letter": [
        {
          "prompt": "[p e r s] &nbsp [p e r s t]<br>[q l g f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[h a z b] &nbsp [h a z b p]<br>[y q l g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[e r s t] &nbsp [e r s t w]<br>[u n o y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[t w m c] &nbsp [t w m c i]<br>[u n o y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[m c i x] &nbsp [m c i x v]<br>[t w m c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[t w m c] &nbsp [t w m c i]<br>[s t w m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[y q l g] &nbsp [y q l g f]<br>[m c i x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[u n o y] &nbsp [u n o y q]<br>[j d k h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[k h a z] &nbsp [k h a z b]<br>[s t w m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[u n o y] &nbsp [u n o y q]<br>[t w m c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "remove_redundant": [
        {
          "prompt": "[h a z b b p] &nbsp [h a z b p]<br>[p p e r s t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[u n o o y q] &nbsp [u n o y q]<br>[z b p e e r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[j d k k h a] &nbsp [j d k h a]<br>[z z b p e r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[f f j d k h] &nbsp [f j d k h]<br>[b p e r r s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[m c i x x v] &nbsp [m c i x v]<br>[u u n o y q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[m c i i x v] &nbsp [m c i x v]<br>[q l l g f j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[p e r r s t] &nbsp [p e r s t]<br>[l l g f j d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[u n o y q q] &nbsp [u n o y q]<br>[r s t w m m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[w m c i x x] &nbsp [w m c i x]<br>[g f f j d k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[u u n o y q] &nbsp [u n o y q]<br>[z z b p e r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "fix_alphabet": [
        {
          "prompt": "[l g f u d] &nbsp [l g f j d]<br>[i y q l g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[w m c i a] &nbsp [w m c i x]<br>[k x a z b] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[w o y q l] &nbsp [n o y q l]<br>[m c d x v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[m c i r v] &nbsp [m c i x v]<br>[a c b p e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[q l g m j] &nbsp [q l g f j]<br>[s u w m c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[n o y q x] &nbsp [n o y q l]<br>[u m o y q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[w m j i x] &nbsp [w m c i x]<br>[s d w m c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[u t w m c] &nbsp [s t w m c]<br>[z b d e r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[p e r f t] &nbsp [p e r s t]<br>[w m c i k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[q l g v j] &nbsp [q l g f j]<br>[u b p e r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "sort": [
        {
          "prompt": "[q o y n l] &nbsp [n o y q l]<br>[v c i x m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[l q y g f] &nbsp [y q l g f]<br>[t w m i c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[w i c m x] &nbsp [w m c i x]<br>[u n q y o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[p b z e r] &nbsp [z b p e r]<br>[m c x i v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[v c i x m] &nbsp [m c i x v]<br>[f q l g y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[n q y o l] &nbsp [n o y q l]<br>[c w m t i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[y q l f g] &nbsp [y q l g f]<br>[j f g d k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[q l f g j] &nbsp [q l g f j]<br>[j d k a h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[f j d h k] &nbsp [f j d k h]<br>[r s m w t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[a b z p e] &nbsp [a z b p e]<br>[q l j f g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "attn": [
        {
          "prompt": "[o y q l] &nbsp [o y q g]<br>[t w m c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[g f j d] &nbsp [g f j k]<br>[m c i x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[l g f j] &nbsp [l g f d]<br>[g f j d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[t w m c] &nbsp [t w m i]<br>[p e r s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[b p e r] &nbsp [b p e s]<br>[m c i x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[a z b p] &nbsp [a z b e]<br>[u n o y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[u n o y] &nbsp [u n o q]<br>[b p e r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[s t w m] &nbsp [s t w c]<br>[p e r s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[u n o y] &nbsp [u n o q]<br>[p e r s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[q l g f] &nbsp [q l g j]<br>[m c i x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ]
    },
    "alph_1": {
      "shuffled_letters": [
        "g",
        "z",
        "y",
        "w",
        "m",
        "r",
        "d",
        "t",
        "s",
        "j",
        "e",
        "o",
        "f",
        "l",
        "u",
        "v",
        "p",
        "n",
        "q",
        "c"
      ],
      "shuffled_alphabet": [
        "a",
        "b",
        "g",
        "z",
        "y",
        "w",
        "m",
        "h",
        "i",
        "r",
        "k",
        "d",
        "t",
        "s",
        "j",
        "e",
        "o",
        "f",
        "l",
        "u",
        "v",
        "p",
        "n",
        "x",
        "q",
        "c"
      ],
      "succ": [
        {
          "prompt": "[s j e o] &nbsp [s j e f]<br>[t s j e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[p n x q] &nbsp [p n x c]<br>[y w m h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[p n x q] &nbsp [p n x c]<br>[g z y w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[k d t s] &nbsp [k d t j]<br>[h i r k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[v p n x] &nbsp [v p n q]<br>[b g z y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[p n x q] &nbsp [p n x c]<br>[z y w m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[s j e o] &nbsp [s j e f]<br>[w m h i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[a b g z] &nbsp [a b g y]<br>[f l u v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[p n x q] &nbsp [p n x c]<br>[f l u v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[g z y w] &nbsp [g z y m]<br>[a b g z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "pred": [
        {
          "prompt": "[d t s j] &nbsp [k t s j]<br>[z y w m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[m h i r] &nbsp [w h i r]<br>[l u v p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[m h i r] &nbsp [w h i r]<br>[w m h i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[s j e o] &nbsp [t j e o]<br>[z y w m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[r k d t] &nbsp [i k d t]<br>[y w m h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[m h i r] &nbsp [w h i r]<br>[y w m h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[s j e o] &nbsp [t j e o]<br>[g z y w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[r k d t] &nbsp [i k d t]<br>[n x q c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[b g z y] &nbsp [a g z y]<br>[m h i r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[i r k d] &nbsp [h r k d]<br>[f l u v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "add_letter": [
        {
          "prompt": "[u v p n] &nbsp [u v p n x]<br>[e o f l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[r k d t] &nbsp [r k d t s]<br>[v p n x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[w m h i] &nbsp [w m h i r]<br>[p n x q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[y w m h] &nbsp [y w m h i]<br>[f l u v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[f l u v] &nbsp [f l u v p]<br>[m h i r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[y w m h] &nbsp [y w m h i]<br>[r k d t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[l u v p] &nbsp [l u v p n]<br>[p n x q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[t s j e] &nbsp [t s j e o]<br>[o f l u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[p n x q] &nbsp [p n x q c]<br>[l u v p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[k d t s] &nbsp [k d t s j]<br>[w m h i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "remove_redundant": [
        {
          "prompt": "[a a b g z y] &nbsp [a b g z y]<br>[e o o f l u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[m h i r r k] &nbsp [m h i r k]<br>[w m h i r r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[o f l u u v] &nbsp [o f l u v]<br>[p n x x q c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[e o f l u u] &nbsp [e o f l u]<br>[u v v p n x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[p n x q c c] &nbsp [p n x q c]<br>[g z z y w m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[g z y y w m] &nbsp [g z y w m]<br>[m h i i r k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[j e o o f l] &nbsp [j e o f l]<br>[f l u v v p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[s j e o f f] &nbsp [s j e o f]<br>[d t s j j e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[s j e o f f] &nbsp [s j e o f]<br>[m m h i r k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[m h i r r k] &nbsp [m h i r k]<br>[v v p n x q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "fix_alphabet": [
        {
          "prompt": "[e o n l u] &nbsp [e o f l u]<br>[c t s j e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[g z y w q] &nbsp [g z y w m]<br>[w e o f l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[l c v p n] &nbsp [l u v p n]<br>[r k b t s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[m h i f k] &nbsp [m h i r k]<br>[j e o f w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[f l u v b] &nbsp [f l u v p]<br>[b g z x w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[f h i r k] &nbsp [m h i r k]<br>[o f l p v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[z y b m h] &nbsp [z y w m h]<br>[w m h i q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[w m l i r] &nbsp [w m h i r]<br>[a r k d t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[d t s j m] &nbsp [d t s j e]<br>[d j e o f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[k v p n x] &nbsp [u v p n x]<br>[p n x y c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "sort": [
        {
          "prompt": "[e f o l u] &nbsp [e o f l u]<br>[h y w m z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[y b g z a] &nbsp [a b g z y]<br>[k j t s d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[i w m h y] &nbsp [y w m h i]<br>[e o u l f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[q p n x v] &nbsp [v p n x q]<br>[j s t e o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[h w m y i] &nbsp [y w m h i]<br>[v l u f p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[n p x q c] &nbsp [p n x q c]<br>[e o f u l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[e o u l f] &nbsp [e o f l u]<br>[f u l v p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[h i k r d] &nbsp [h i r k d]<br>[v q n x p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[o j e s f] &nbsp [s j e o f]<br>[l u v n p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[w h m i r] &nbsp [w m h i r]<br>[v n p x q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "attn": [
        {
          "prompt": "[p n x q] &nbsp [p n x c]<br>[z y w m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[p n x q] &nbsp [p n x c]<br>[b g z y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[y w m h] &nbsp [y w m i]<br>[p n x q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[r k d t] &nbsp [r k d s]<br>[j e o f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[r k d t] &nbsp [r k d s]<br>[a b g z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[p n x q] &nbsp [p n x c]<br>[i r k d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[w m h i] &nbsp [w m h r]<br>[r k d t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[g z y w] &nbsp [g z y m]<br>[j e o f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[v p n x] &nbsp [v p n q]<br>[l u v p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[z y w m] &nbsp [z y w h]<br>[m h i r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ]
    },
    "alph_2": {
      "shuffled_letters": [
        "b",
        "n",
        "f",
        "m",
        "p",
        "g",
        "h",
        "w",
        "q",
        "e",
        "d",
        "l",
        "y",
        "s",
        "o",
        "c",
        "k",
        "a",
        "r",
        "j"
      ],
      "shuffled_alphabet": [
        "b",
        "n",
        "f",
        "m",
        "p",
        "g",
        "h",
        "w",
        "i",
        "q",
        "e",
        "d",
        "l",
        "y",
        "s",
        "o",
        "c",
        "k",
        "a",
        "t",
        "u",
        "v",
        "r",
        "x",
        "j",
        "z"
      ],
      "succ": [
        {
          "prompt": "[s o c k] &nbsp [s o c a]<br>[f m p g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[q e d l] &nbsp [q e d y]<br>[k a t u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[q e d l] &nbsp [q e d y]<br>[n f m p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[l y s o] &nbsp [l y s c]<br>[e d l y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[l y s o] &nbsp [l y s c]<br>[n f m p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[c k a t] &nbsp [c k a u]<br>[y s o c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[c k a t] &nbsp [c k a u]<br>[k a t u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[f m p g] &nbsp [f m p h]<br>[b n f m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[g h w i] &nbsp [g h w q]<br>[d l y s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[f m p g] &nbsp [f m p h]<br>[u v r x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "pred": [
        {
          "prompt": "[s o c k] &nbsp [y o c k]<br>[n f m p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[e d l y] &nbsp [q d l y]<br>[y s o c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[c k a t] &nbsp [o k a t]<br>[a t u v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[l y s o] &nbsp [d y s o]<br>[y s o c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[l y s o] &nbsp [d y s o]<br>[t u v r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[a t u v] &nbsp [k t u v]<br>[h w i q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[v r x j] &nbsp [u r x j]<br>[o c k a] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[l y s o] &nbsp [d y s o]<br>[n f m p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[w i q e] &nbsp [h i q e]<br>[q e d l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[o c k a] &nbsp [s c k a]<br>[a t u v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "add_letter": [
        {
          "prompt": "[g h w i] &nbsp [g h w i q]<br>[a t u v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[f m p g] &nbsp [f m p g h]<br>[a t u v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[q e d l] &nbsp [q e d l y]<br>[a t u v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[m p g h] &nbsp [m p g h w]<br>[n f m p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[d l y s] &nbsp [d l y s o]<br>[v r x j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[n f m p] &nbsp [n f m p g]<br>[l y s o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[l y s o] &nbsp [l y s o c]<br>[e d l y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[n f m p] &nbsp [n f m p g]<br>[b n f m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[m p g h] &nbsp [m p g h w]<br>[g h w i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[i q e d] &nbsp [i q e d l]<br>[b n f m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "remove_redundant": [
        {
          "prompt": "[s o c k k a] &nbsp [s o c k a]<br>[m m p g h w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[m p g g h w] &nbsp [m p g h w]<br>[a a t u v r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[o c c k a t] &nbsp [o c k a t]<br>[v r x x j z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[h h w i q e] &nbsp [h w i q e]<br>[v r x j z z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[k a a t u v] &nbsp [k a t u v]<br>[f m p g g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[w i q e e d] &nbsp [w i q e d]<br>[m m p g h w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[e d l y s s] &nbsp [e d l y s]<br>[b n f f m p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[v r x j z z] &nbsp [v r x j z]<br>[w i q e e d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[w i q q e d] &nbsp [w i q e d]<br>[v v r x j z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[h w i q q e] &nbsp [h w i q e]<br>[b n f m m p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "fix_alphabet": [
        {
          "prompt": "[f m p g o] &nbsp [f m p g h]<br>[l y s r c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[l y f o c] &nbsp [l y s o c]<br>[x i q e d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[o c x a t] &nbsp [o c k a t]<br>[a t q v r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[a m p g h] &nbsp [f m p g h]<br>[j k a t u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[p g z w i] &nbsp [p g h w i]<br>[a m p g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[v a x j z] &nbsp [v r x j z]<br>[l y s p c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[n a m p g] &nbsp [n f m p g]<br>[w i q a d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[s r c k a] &nbsp [s o c k a]<br>[y s h c k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[s o c j a] &nbsp [s o c k a]<br>[m p g h r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[b n f j p] &nbsp [b n f m p]<br>[p g h t i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "sort": [
        {
          "prompt": "[h g p w i] &nbsp [p g h w i]<br>[s k c o a] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[b f n m p] &nbsp [b n f m p]<br>[a k t u v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[w i e q d] &nbsp [w i q e d]<br>[f m p h g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[v u t r x] &nbsp [t u v r x]<br>[u v r j x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[s d l y e] &nbsp [e d l y s]<br>[w e q i d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[f n m p g] &nbsp [n f m p g]<br>[o c t a k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[b n f p m] &nbsp [b n f m p]<br>[w h g i q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[t x v r u] &nbsp [t u v r x]<br>[m f n p g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[n p m f g] &nbsp [n f m p g]<br>[p i h w g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[o c a k t] &nbsp [o c k a t]<br>[w d q e i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "attn": [
        {
          "prompt": "[s o c k] &nbsp [s o c a]<br>[o c k a] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[m p g h] &nbsp [m p g w]<br>[q e d l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[y s o c] &nbsp [y s o k]<br>[m p g h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[v r x j] &nbsp [v r x z]<br>[b n f m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[y s o c] &nbsp [y s o k]<br>[d l y s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[w i q e] &nbsp [w i q d]<br>[b n f m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[o c k a] &nbsp [o c k t]<br>[c k a t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[n f m p] &nbsp [n f m g]<br>[a t u v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[h w i q] &nbsp [h w i e]<br>[v r x j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[u v r x] &nbsp [u v r j]<br>[b n f m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ]
    },
    "alph_3": {
      "shuffled_letters": [
        "i",
        "s",
        "f",
        "z",
        "v",
        "t",
        "c",
        "r",
        "o",
        "g",
        "e",
        "w",
        "j",
        "m",
        "x",
        "q",
        "n",
        "y",
        "a",
        "k"
      ],
      "shuffled_alphabet": [
        "i",
        "b",
        "s",
        "d",
        "f",
        "z",
        "v",
        "h",
        "t",
        "c",
        "r",
        "l",
        "o",
        "g",
        "e",
        "p",
        "w",
        "j",
        "m",
        "x",
        "u",
        "q",
        "n",
        "y",
        "a",
        "k"
      ],
      "succ": [
        {
          "prompt": "[p w j m] &nbsp [p w j x]<br>[l o g e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[x u q n] &nbsp [x u q y]<br>[w j m x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[i b s d] &nbsp [i b s f]<br>[p w j m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[q n y a] &nbsp [q n y k]<br>[u q n y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[e p w j] &nbsp [e p w m]<br>[g e p w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[z v h t] &nbsp [z v h c]<br>[l o g e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[h t c r] &nbsp [h t c l]<br>[m x u q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[i b s d] &nbsp [i b s f]<br>[l o g e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[q n y a] &nbsp [q n y k]<br>[w j m x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[s d f z] &nbsp [s d f v]<br>[i b s d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "pred": [
        {
          "prompt": "[h t c r] &nbsp [v t c r]<br>[v h t c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[x u q n] &nbsp [m u q n]<br>[l o g e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[g e p w] &nbsp [o e p w]<br>[c r l o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[r l o g] &nbsp [c l o g]<br>[v h t c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[q n y a] &nbsp [u n y a]<br>[d f z v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[q n y a] &nbsp [u n y a]<br>[n y a k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[u q n y] &nbsp [x q n y]<br>[s d f z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[z v h t] &nbsp [f v h t]<br>[x u q n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[v h t c] &nbsp [z h t c]<br>[n y a k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[d f z v] &nbsp [s f z v]<br>[v h t c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "add_letter": [
        {
          "prompt": "[v h t c] &nbsp [v h t c r]<br>[q n y a] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[g e p w] &nbsp [g e p w j]<br>[s d f z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[q n y a] &nbsp [q n y a k]<br>[h t c r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[u q n y] &nbsp [u q n y a]<br>[q n y a] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[d f z v] &nbsp [d f z v h]<br>[j m x u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[m x u q] &nbsp [m x u q n]<br>[q n y a] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[o g e p] &nbsp [o g e p w]<br>[w j m x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[d f z v] &nbsp [d f z v h]<br>[w j m x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[b s d f] &nbsp [b s d f z]<br>[j m x u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[q n y a] &nbsp [q n y a k]<br>[r l o g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "remove_redundant": [
        {
          "prompt": "[h h t c r l] &nbsp [h t c r l]<br>[b s d d f z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[z v h t t c] &nbsp [z v h t c]<br>[f z v h t t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[q n y y a k] &nbsp [q n y a k]<br>[z z v h t c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[w j m x u u] &nbsp [w j m x u]<br>[q n y y a k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[m x u q q n] &nbsp [m x u q n]<br>[q q n y a k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[i i b s d f] &nbsp [i b s d f]<br>[c c r l o g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[w j m x u u] &nbsp [w j m x u]<br>[q n y a a k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[f z v v h t] &nbsp [f z v h t]<br>[l o g g e p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[u q n y y a] &nbsp [u q n y a]<br>[t c c r l o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[g g e p w j] &nbsp [g e p w j]<br>[v h t c c r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "fix_alphabet": [
        {
          "prompt": "[o z v h t] &nbsp [f z v h t]<br>[s w f z v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[j m x u a] &nbsp [j m x u q]<br>[c r d o g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[g w j m x] &nbsp [p w j m x]<br>[y d f z v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[r l o g u] &nbsp [r l o g e]<br>[q n y a s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[z v i t c] &nbsp [z v h t c]<br>[j m x u e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[s w j m x] &nbsp [p w j m x]<br>[e u q n y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[z v m t c] &nbsp [z v h t c]<br>[f q v h t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[g x p w j] &nbsp [g e p w j]<br>[w l o g e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[r l i g e] &nbsp [r l o g e]<br>[q n o a k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[b o g e p] &nbsp [l o g e p]<br>[c a l o g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "sort": [
        {
          "prompt": "[z v t h c] &nbsp [z v h t c]<br>[w e p g j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[v h t r c] &nbsp [v h t c r]<br>[f z v t h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[l o e g p] &nbsp [l o g e p]<br>[a q n y u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[d f h v z] &nbsp [d f z v h]<br>[j m q u x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[i d s b f] &nbsp [i b s d f]<br>[x w j m p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[p x j m w] &nbsp [p w j m x]<br>[i f s d b] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[m n u q x] &nbsp [m x u q n]<br>[g r l o c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[p e w j m] &nbsp [e p w j m]<br>[i b d s f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[x u q y n] &nbsp [x u q n y]<br>[m w j p x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[e p m j w] &nbsp [e p w j m]<br>[t v h z c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "attn": [
        {
          "prompt": "[h t c r] &nbsp [h t c l]<br>[c r l o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[m x u q] &nbsp [m x u n]<br>[i b s d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[j m x u] &nbsp [j m x q]<br>[i b s d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[g e p w] &nbsp [g e p j]<br>[q n y a] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[x u q n] &nbsp [x u q y]<br>[i b s d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[g e p w] &nbsp [g e p j]<br>[v h t c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[l o g e] &nbsp [l o g p]<br>[z v h t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[b s d f] &nbsp [b s d z]<br>[c r l o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[v h t c] &nbsp [v h t r]<br>[t c r l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[x u q n] &nbsp [x u q y]<br>[s d f z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ]
    },
    "alph_4": {
      "shuffled_letters": [
        "w",
        "p",
        "z",
        "b",
        "a",
        "s",
        "j",
        "g",
        "x",
        "o",
        "m",
        "n",
        "t",
        "k",
        "f",
        "i",
        "q",
        "y",
        "r",
        "l"
      ],
      "shuffled_alphabet": [
        "w",
        "p",
        "c",
        "d",
        "e",
        "z",
        "b",
        "h",
        "a",
        "s",
        "j",
        "g",
        "x",
        "o",
        "m",
        "n",
        "t",
        "k",
        "f",
        "i",
        "u",
        "v",
        "q",
        "y",
        "r",
        "l"
      ],
      "succ": [
        {
          "prompt": "[w p c d] &nbsp [w p c e]<br>[k f i u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[n t k f] &nbsp [n t k i]<br>[h a s j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[e z b h] &nbsp [e z b a]<br>[m n t k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[e z b h] &nbsp [e z b a]<br>[f i u v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[x o m n] &nbsp [x o m t]<br>[s j g x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[f i u v] &nbsp [f i u q]<br>[v q y r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[o m n t] &nbsp [o m n k]<br>[n t k f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[b h a s] &nbsp [b h a j]<br>[j g x o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[w p c d] &nbsp [w p c e]<br>[j g x o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[g x o m] &nbsp [g x o n]<br>[i u v q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "pred": [
        {
          "prompt": "[t k f i] &nbsp [n k f i]<br>[d e z b] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[u v q y] &nbsp [i v q y]<br>[f i u v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[z b h a] &nbsp [e b h a]<br>[s j g x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[j g x o] &nbsp [s g x o]<br>[s j g x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[f i u v] &nbsp [k i u v]<br>[m n t k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[x o m n] &nbsp [g o m n]<br>[g x o m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[c d e z] &nbsp [p d e z]<br>[o m n t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[h a s j] &nbsp [b a s j]<br>[a s j g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[o m n t] &nbsp [x m n t]<br>[u v q y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[a s j g] &nbsp [h s j g]<br>[c d e z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "add_letter": [
        {
          "prompt": "[s j g x] &nbsp [s j g x o]<br>[w p c d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[t k f i] &nbsp [t k f i u]<br>[a s j g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[a s j g] &nbsp [a s j g x]<br>[d e z b] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[h a s j] &nbsp [h a s j g]<br>[n t k f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[t k f i] &nbsp [t k f i u]<br>[v q y r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[v q y r] &nbsp [v q y r l]<br>[d e z b] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[a s j g] &nbsp [a s j g x]<br>[e z b h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[s j g x] &nbsp [s j g x o]<br>[c d e z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[z b h a] &nbsp [z b h a s]<br>[w p c d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[v q y r] &nbsp [v q y r l]<br>[c d e z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "remove_redundant": [
        {
          "prompt": "[v v q y r l] &nbsp [v q y r l]<br>[p c d e e z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[a s s j g x] &nbsp [a s j g x]<br>[i u v v q y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[d e z z b h] &nbsp [d e z b h]<br>[s j g x x o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[d e e z b h] &nbsp [d e z b h]<br>[h a s j g g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[v q y r r l] &nbsp [v q y r l]<br>[f i u u v q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[p c c d e z] &nbsp [p c d e z]<br>[d e z b b h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[s s j g x o] &nbsp [s j g x o]<br>[k f i u v v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[i u u v q y] &nbsp [i u v q y]<br>[m n t k k f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[m m n t k f] &nbsp [m n t k f]<br>[s j g x o o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[w w p c d e] &nbsp [w p c d e]<br>[x o m n n t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "fix_alphabet": [
        {
          "prompt": "[f i h v q] &nbsp [f i u v q]<br>[j g x o b] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[z b h l s] &nbsp [z b h a s]<br>[i u v q j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[f e u v q] &nbsp [f i u v q]<br>[d e z b y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[j g x y m] &nbsp [j g x o m]<br>[q t k f i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[l a s j g] &nbsp [h a s j g]<br>[w p c d z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[a s f g x] &nbsp [a s j g x]<br>[a k f i u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[g x o e n] &nbsp [g x o m n]<br>[m n b k f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[t k j i u] &nbsp [t k f i u]<br>[z b d a s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[e z f h a] &nbsp [e z b h a]<br>[j g x e m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[m a t k f] &nbsp [m n t k f]<br>[h a s o g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "sort": [
        {
          "prompt": "[n x o m g] &nbsp [g x o m n]<br>[o m t n k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[j g m o x] &nbsp [j g x o m]<br>[f n t k m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[p e d c z] &nbsp [p c d e z]<br>[m f t k n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[g o x m n] &nbsp [g x o m n]<br>[v q r y l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[g m o x n] &nbsp [g x o m n]<br>[m f t k n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[j g m o x] &nbsp [j g x o m]<br>[q u v i y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[w p c e d] &nbsp [w p c d e]<br>[m o x n t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[g x n m o] &nbsp [g x o m n]<br>[o m k t n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[o g x j m] &nbsp [j g x o m]<br>[u f i k v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[g x o n m] &nbsp [g x o m n]<br>[c p w d e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "attn": [
        {
          "prompt": "[t k f i] &nbsp [t k f u]<br>[a s j g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[t k f i] &nbsp [t k f u]<br>[n t k f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[b h a s] &nbsp [b h a j]<br>[v q y r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[f i u v] &nbsp [f i u q]<br>[s j g x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[m n t k] &nbsp [m n t f]<br>[k f i u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[w p c d] &nbsp [w p c e]<br>[a s j g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[a s j g] &nbsp [a s j x]<br>[x o m n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[e z b h] &nbsp [e z b a]<br>[a s j g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[e z b h] &nbsp [e z b a]<br>[b h a s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[d e z b] &nbsp [d e z h]<br>[c d e z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ]
    },
    "alph_5": {
      "shuffled_letters": [
        "r",
        "y",
        "t",
        "b",
        "n",
        "x",
        "s",
        "z",
        "p",
        "d",
        "u",
        "v",
        "o",
        "g",
        "i",
        "k",
        "h",
        "l",
        "w",
        "c"
      ],
      "shuffled_alphabet": [
        "a",
        "r",
        "y",
        "t",
        "e",
        "f",
        "b",
        "n",
        "x",
        "j",
        "s",
        "z",
        "m",
        "p",
        "d",
        "u",
        "q",
        "v",
        "o",
        "g",
        "i",
        "k",
        "h",
        "l",
        "w",
        "c"
      ],
      "succ": [
        {
          "prompt": "[a r y t] &nbsp [a r y e]<br>[n x j s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[n x j s] &nbsp [n x j z]<br>[a r y t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[z m p d] &nbsp [z m p u]<br>[x j s z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[o g i k] &nbsp [o g i h]<br>[m p d u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[r y t e] &nbsp [r y t f]<br>[k h l w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[u q v o] &nbsp [u q v g]<br>[k h l w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[z m p d] &nbsp [z m p u]<br>[u q v o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[e f b n] &nbsp [e f b x]<br>[q v o g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[b n x j] &nbsp [b n x s]<br>[d u q v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[k h l w] &nbsp [k h l c]<br>[u q v o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "pred": [
        {
          "prompt": "[z m p d] &nbsp [s m p d]<br>[m p d u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[y t e f] &nbsp [r t e f]<br>[k h l w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[h l w c] &nbsp [k l w c]<br>[j s z m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[b n x j] &nbsp [f n x j]<br>[m p d u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[p d u q] &nbsp [m d u q]<br>[h l w c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[p d u q] &nbsp [m d u q]<br>[s z m p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[y t e f] &nbsp [r t e f]<br>[f b n x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[m p d u] &nbsp [z p d u]<br>[f b n x] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[o g i k] &nbsp [v g i k]<br>[n x j s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[g i k h] &nbsp [o i k h]<br>[b n x j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "add_letter": [
        {
          "prompt": "[a r y t] &nbsp [a r y t e]<br>[j s z m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[f b n x] &nbsp [f b n x j]<br>[b n x j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[e f b n] &nbsp [e f b n x]<br>[k h l w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[a r y t] &nbsp [a r y t e]<br>[d u q v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[q v o g] &nbsp [q v o g i]<br>[m p d u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[e f b n] &nbsp [e f b n x]<br>[v o g i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[k h l w] &nbsp [k h l w c]<br>[j s z m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[r y t e] &nbsp [r y t e f]<br>[b n x j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[j s z m] &nbsp [j s z m p]<br>[u q v o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[v o g i] &nbsp [v o g i k]<br>[e f b n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "remove_redundant": [
        {
          "prompt": "[a r y y t e] &nbsp [a r y t e]<br>[b n n x j s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[b b n x j s] &nbsp [b n x j s]<br>[m p p d u q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[q v o o g i] &nbsp [q v o g i]<br>[v v o g i k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[b n n x j s] &nbsp [b n x j s]<br>[x j j s z m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[p d u q v v] &nbsp [p d u q v]<br>[y t e f f b] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[p d d u q v] &nbsp [p d u q v]<br>[r y y t e f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[x j s z z m] &nbsp [x j s z m]<br>[o g i k k h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[p d u q q v] &nbsp [p d u q v]<br>[k h l w w c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[b b n x j s] &nbsp [b n x j s]<br>[x j s s z m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[j j s z m p] &nbsp [j s z m p]<br>[a a r y t e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "fix_alphabet": [
        {
          "prompt": "[g e f b n] &nbsp [t e f b n]<br>[y t e f j] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[z m p x u] &nbsp [z m p d u]<br>[i n x j s] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[v q g i k] &nbsp [v o g i k]<br>[m r y t e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[w q v o g] &nbsp [u q v o g]<br>[y d e f b] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[k h l t c] &nbsp [k h l w c]<br>[c k h l w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[b o x j s] &nbsp [b n x j s]<br>[n w j s z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[z m p d f] &nbsp [z m p d u]<br>[y t e i b] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[e f b n z] &nbsp [e f b n x]<br>[z m b d u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[u q v s g] &nbsp [u q v o g]<br>[p d b q v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[v b n x j] &nbsp [f b n x j]<br>[k h l a c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "sort": [
        {
          "prompt": "[q v g o i] &nbsp [q v o g i]<br>[p z m s d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[g v o q i] &nbsp [q v o g i]<br>[s d m p z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[q o v g i] &nbsp [q v o g i]<br>[x f b n e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[e r y t a] &nbsp [a r y t e]<br>[z s j m p] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[z s m p d] &nbsp [s z m p d]<br>[t y r e f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[n z j s x] &nbsp [n x j s z]<br>[j s z p m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[u v q o g] &nbsp [u q v o g]<br>[b t e f y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[p z m s d] &nbsp [s z m p d]<br>[a r y e t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[n e f b t] &nbsp [t e f b n]<br>[w h l k c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[t b f e n] &nbsp [t e f b n]<br>[n j x s z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "attn": [
        {
          "prompt": "[d u q v] &nbsp [d u q o]<br>[g i k h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[m p d u] &nbsp [m p d q]<br>[e f b n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[j s z m] &nbsp [j s z p]<br>[a r y t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[g i k h] &nbsp [g i k l]<br>[x j s z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[y t e f] &nbsp [y t e b]<br>[x j s z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[v o g i] &nbsp [v o g k]<br>[j s z m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[k h l w] &nbsp [k h l c]<br>[x j s z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[k h l w] &nbsp [k h l c]<br>[r y t e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[z m p d] &nbsp [z m p u]<br>[k h l w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[n x j s] &nbsp [n x j z]<br>[y t e f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ]
    },
    "alph_6": {
      "shuffled_letters": [
        "x",
        "p",
        "j",
        "h",
        "n",
        "a",
        "u",
        "d",
        "q",
        "c",
        "t",
        "f",
        "s",
        "b",
        "g",
        "y",
        "e",
        "z",
        "l",
        "o"
      ],
      "shuffled_alphabet": [
        "x",
        "p",
        "j",
        "h",
        "n",
        "a",
        "u",
        "d",
        "i",
        "q",
        "k",
        "c",
        "m",
        "t",
        "f",
        "s",
        "b",
        "r",
        "g",
        "y",
        "e",
        "v",
        "w",
        "z",
        "l",
        "o"
      ],
      "succ": [
        {
          "prompt": "[m t f s] &nbsp [m t f b]<br>[i q k c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[t f s b] &nbsp [t f s r]<br>[h n a u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[v w z l] &nbsp [v w z o]<br>[c m t f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[f s b r] &nbsp [f s b g]<br>[g y e v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[x p j h] &nbsp [x p j n]<br>[f s b r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[d i q k] &nbsp [d i q c]<br>[x p j h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[p j h n] &nbsp [p j h a]<br>[v w z l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[m t f s] &nbsp [m t f b]<br>[v w z l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[v w z l] &nbsp [v w z o]<br>[p j h n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[j h n a] &nbsp [j h n u]<br>[v w z l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "pred": [
        {
          "prompt": "[j h n a] &nbsp [p h n a]<br>[v w z l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[m t f s] &nbsp [c t f s]<br>[d i q k] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[v w z l] &nbsp [e w z l]<br>[w z l o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[r g y e] &nbsp [b g y e]<br>[b r g y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[t f s b] &nbsp [m f s b]<br>[u d i q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[p j h n] &nbsp [x j h n]<br>[t f s b] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[y e v w] &nbsp [g e v w]<br>[h n a u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[t f s b] &nbsp [m f s b]<br>[e v w z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[n a u d] &nbsp [h a u d]<br>[h n a u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[d i q k] &nbsp [u i q k]<br>[f s b r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "add_letter": [
        {
          "prompt": "[s b r g] &nbsp [s b r g y]<br>[k c m t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[n a u d] &nbsp [n a u d i]<br>[i q k c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[p j h n] &nbsp [p j h n a]<br>[t f s b] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[x p j h] &nbsp [x p j h n]<br>[i q k c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[x p j h] &nbsp [x p j h n]<br>[e v w z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[i q k c] &nbsp [i q k c m]<br>[b r g y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[u d i q] &nbsp [u d i q k]<br>[q k c m] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[q k c m] &nbsp [q k c m t]<br>[e v w z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[m t f s] &nbsp [m t f s b]<br>[x p j h] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[t f s b] &nbsp [t f s b r]<br>[u d i q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "remove_redundant": [
        {
          "prompt": "[y e v w w z] &nbsp [y e v w z]<br>[q k c c m t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[u d d i q k] &nbsp [u d i q k]<br>[v w z l l o] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[p j h h n a] &nbsp [p j h n a]<br>[n n a u d i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[x p p j h n] &nbsp [x p j h n]<br>[m m t f s b] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[d i q k c c] &nbsp [d i q k c]<br>[y e v w w z] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[v v w z l o] &nbsp [v w z l o]<br>[j h h n a u] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[m t f f s b] &nbsp [m t f s b]<br>[b r g y y e] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[n a u d i i] &nbsp [n a u d i]<br>[r g y e v v] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[n a u u d i] &nbsp [n a u d i]<br>[f s b r r g] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[c c m t f s] &nbsp [c m t f s]<br>[d i q q k c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "fix_alphabet": [
        {
          "prompt": "[c m t n s] &nbsp [c m t f s]<br>[q p c m t] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[u d y q k] &nbsp [u d i q k]<br>[m t y s b] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[e v w b l] &nbsp [e v w z l]<br>[c m t f r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[p l h n a] &nbsp [p j h n a]<br>[h n v u d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[x p j h v] &nbsp [x p j h n]<br>[a f d i q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[l j h n a] &nbsp [p j h n a]<br>[n k u d i] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[f s w r g] &nbsp [f s b r g]<br>[d v q k c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[l n a u d] &nbsp [h n a u d]<br>[m t f g b] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[a l d i q] &nbsp [a u d i q]<br>[x p j h r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[i q r c m] &nbsp [i q k c m]<br>[s b r k y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "sort": [
        {
          "prompt": "[d i q c k] &nbsp [d i q k c]<br>[b r g e y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[c s t f m] &nbsp [c m t f s]<br>[d i c k q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[u d i k q] &nbsp [u d i q k]<br>[v w z o l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[a h n j u] &nbsp [j h n a u]<br>[n h a u d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[v w o l z] &nbsp [v w z l o]<br>[k m c t f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[i d u q k] &nbsp [u d i q k]<br>[k t m c f] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[v w l z o] &nbsp [v w z l o]<br>[q k t m c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[l w z v o] &nbsp [v w z l o]<br>[t k c m q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[a u q i d] &nbsp [a u d i q]<br>[v o z l w] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[m s f t b] &nbsp [m t f s b]<br>[u n a h d] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "attn": [
        {
          "prompt": "[v w z l] &nbsp [v w z o]<br>[f s b r] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[h n a u] &nbsp [h n a d]<br>[p j h n] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[p j h n] &nbsp [p j h a]<br>[t f s b] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[u d i q] &nbsp [u d i k]<br>[t f s b] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[p j h n] &nbsp [p j h a]<br>[b r g y] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[d i q k] &nbsp [d i q c]<br>[i q k c] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[s b r g] &nbsp [s b r y]<br>[v w z l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[h n a u] &nbsp [h n a d]<br>[v w z l] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[b r g y] &nbsp [b r g e]<br>[u d i q] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[c m t f] &nbsp [c m t s]<br>[j h n a] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ]
    }
  },
  "np_symb": {
    "symb_0": {
      "shuffled_letters": null,
      "shuffled_alphabet": [
        ">",
        "$",
        ")",
        "*",
        "@",
        "=",
        "&",
        "<",
        "+",
        "!"
      ],
      "succ": [
        {
          "prompt": "[$ ) * @] &nbsp [$ ) * =]<br>[) * @ =] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[@ = & <] &nbsp [@ = & +]<br>[$ ) * @] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[$ ) * @] &nbsp [$ ) * =]<br>[= & < +] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[* @ = &] &nbsp [* @ = <]<br>[> $ ) *] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[@ = & <] &nbsp [@ = & +]<br>[* @ = &] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[* @ = &] &nbsp [* @ = <]<br>[= & < +] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[* @ = &] &nbsp [* @ = <]<br>[@ = & <] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[= & < +] &nbsp [= & < !]<br>[) * @ =] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[* @ = &] &nbsp [* @ = <]<br>[) * @ =] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[> $ ) *] &nbsp [> $ ) @]<br>[) * @ =] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "pred": [
        {
          "prompt": "[$ ) * @] &nbsp [> ) * @]<br>[) * @ =] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[) * @ =] &nbsp [$ * @ =]<br>[* @ = &] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[* @ = &] &nbsp [) @ = &]<br>[= & < +] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[* @ = &] &nbsp [) @ = &]<br>[@ = & <] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[$ ) * @] &nbsp [> ) * @]<br>[@ = & <] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[& < + !] &nbsp [= < + !]<br>[* @ = &] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[@ = & <] &nbsp [* = & <]<br>[* @ = &] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[$ ) * @] &nbsp [> ) * @]<br>[* @ = &] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[& < + !] &nbsp [= < + !]<br>[) * @ =] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[@ = & <] &nbsp [* = & <]<br>[= & < +] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ]
    },
    "symb_1": {
      "shuffled_letters": null,
      "shuffled_alphabet": [
        "&",
        ">",
        "$",
        "!",
        "+",
        ")",
        "<",
        "@",
        "=",
        "*"
      ],
      "succ": [
        {
          "prompt": "[) < @ =] &nbsp [) < @ *]<br>[+ ) < @] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[> $ ! +] &nbsp [> $ ! )]<br>[& > $ !] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[$ ! + )] &nbsp [$ ! + <]<br>[> $ ! +] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[! + ) <] &nbsp [! + ) @]<br>[+ ) < @] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[> $ ! +] &nbsp [> $ ! )]<br>[+ ) < @] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[$ ! + )] &nbsp [$ ! + <]<br>[) < @ =] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[! + ) <] &nbsp [! + ) @]<br>[> $ ! +] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[) < @ =] &nbsp [) < @ *]<br>[$ ! + )] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[& > $ !] &nbsp [& > $ +]<br>[$ ! + )] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[& > $ !] &nbsp [& > $ +]<br>[! + ) <] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ],
      "pred": [
        {
          "prompt": "[+ ) < @] &nbsp [! ) < @]<br>[) < @ =] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 0
        },
        {
          "prompt": "[$ ! + )] &nbsp [> ! + )]<br>[+ ) < @] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 1
        },
        {
          "prompt": "[< @ = *] &nbsp [) @ = *]<br>[$ ! + )] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 2
        },
        {
          "prompt": "[$ ! + )] &nbsp [> ! + )]<br>[> $ ! +] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 3
        },
        {
          "prompt": "[+ ) < @] &nbsp [! ) < @]<br>[! + ) <] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 4
        },
        {
          "prompt": "[) < @ =] &nbsp [+ < @ =]<br>[+ ) < @] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 5
        },
        {
          "prompt": "[! + ) <] &nbsp [$ + ) <]<br>[< @ = *] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 6
        },
        {
          "prompt": "[! + ) <] &nbsp [$ + ) <]<br>[> $ ! +] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 7
        },
        {
          "prompt": "[! + ) <] &nbsp [$ + ) <]<br>[$ ! + )] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 8
        },
        {
          "prompt": "[< @ = *] &nbsp [) @ = *]<br>[> $ ! +] &nbsp [&nbsp ? &nbsp]",
          "prob_ind": 9
        }
      ]
    }
  }
}