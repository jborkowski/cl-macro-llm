;;; tests.lisp — generated from j14i/cl-ds row 140
;;; instruction: Write a Common Lisp macro `with-collector` that hides list head and tail pointers and exposes a user-named local functio
;;; category: capture-management | technique: closure-capture | complexity: advanced
;;; quality_score: 1.0

(((with-collector (collect) (collect 1) (collect 2)) . (LET ((#:G1 NIL) (#:G2 NIL)) (FLET ((COLLECT (#:G3) (LET ((#:G4 (CONS #:G3 NIL))) (IF #:G1 (SETF (CDR #:G2) #:G4 #:G2 #:G4) (SETF #:G1 #:G4 #:G2 #:G4))))) (COLLECT 1) (COLLECT 2) #:G1))))
