;;; tests.lisp — generated from j14i/cl-ds row 46
;;; instruction: Write a Common Lisp macro `for-i-from-to` that iterates a user-named variable from start through end inclusive. The end 
;;; category: capture-management | technique: gensym | complexity: intermediate
;;; quality_score: 1.0

(((for-i-from-to i 0 10 (print i)) . (LET ((#:G1 10)) (DO ((I 0 (1+ I))) ((> I #:G1)) (PRINT I)))))
