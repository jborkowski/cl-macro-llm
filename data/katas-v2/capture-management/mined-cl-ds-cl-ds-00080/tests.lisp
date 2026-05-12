;;; tests.lisp — generated from j14i/cl-ds row 80
;;; instruction: Write a Common Lisp macro to handle this code pattern:
;;; category: capture-management | technique: once-only,gensym | complexity: intermediate
;;; quality_score: 0.875

(((is-print (princ "ABCDEFGH") "ABCDEFGHIJKLMNO") . (let ((#:expected203 "ABCDEFGHIJKLMNO") (#:desc204 nil)) (with-catching-errors (:description #:desc204 :expected #:expected203) (let* (#:duration201 (#:output200 (with-output-to-string (*standard-output*) (with-duration ((#:duration-inner202 #:output200) (princ "ABCDEFGH")) (declare (ignore #:output200)) (setq #:duration201 #:duration-inner202))))) (test #:output200 #:expected203 #:desc204 :duration #:duration201 :got-form (quote (princ "ABCDEFGH")) :test-fn (function string=) :report-expected-label "output"))))))
