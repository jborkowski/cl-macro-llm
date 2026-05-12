;;; tests.lisp — generated from j14i/cl-ds row 143
;;; instruction: Write a Common Lisp macro `do-pairs` that walks a list two elements at a time, binding each pair to user-named symbols. 
;;; category: capture-management | technique: gensym | complexity: intermediate
;;; quality_score: 1.0

(((do-pairs (k v lst) (print (cons k v))) . (LET ((#:G1 LST)) (LOOP WHILE #:G1 DO (LET ((K (POP #:G1)) (V (POP #:G1))) (PRINT (CONS K V)))))))
