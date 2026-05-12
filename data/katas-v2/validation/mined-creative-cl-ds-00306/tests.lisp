;;; tests.lisp — generated from j14i/cl-ds row 306
;;; instruction: Write a Common Lisp macro `validate-plist` that evaluates a plist once and signals an error if any of the required keys 
;;; category: validation | technique: gensym | complexity: intermediate
;;; quality_score: 1.0

(((validate-plist opts :host :port) . (LET ((#:G1 OPTS)) (UNLESS (GETF #:G1 :HOST) (ERROR "Missing key ~S" :HOST)) (UNLESS (GETF #:G1 :PORT) (ERROR "Missing key ~S" :PORT)) #:G1)))
