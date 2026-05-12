;;; tests.lisp — generated from j14i/cl-ds row 244
;;; instruction: Write a Common Lisp macro `profile-block` that times a labelled body in real time, prints the elapsed seconds to *trace-
;;; category: debugging | technique: gensym | complexity: intermediate
;;; quality_score: 1.0

(((profile-block "sort" (sort-data items)) . (LET ((#:G1 (GET-INTERNAL-REAL-TIME))) (LET ((#:G2 (PROGN (SORT-DATA ITEMS)))) (FORMAT *TRACE-OUTPUT* "~&[~A] ~F seconds~%" "sort" (/ (- (GET-INTERNAL-REAL-TIME) #:G1) INTERNAL-TIME-UNITS-PER-SECOND)) #:G2)) ))
