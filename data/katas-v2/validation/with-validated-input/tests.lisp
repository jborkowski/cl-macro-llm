;;; tests.lisp — hand-crafted, CL idiom
;;; instruction: Write `with-validated-input` — bind, validate via a
;;; predicate, signal on failure, then run body.
;;; category: validation | technique: bind-and-validate | complexity: intermediate
;;; quality_score: 0.95

(
 ((with-validated-input (n (read-int) plusp) (* n 2))
  . (LET ((N (READ-INT)))
      (UNLESS (PLUSP N) (ERROR "invalid input: ~A" N))
      (* N 2)))

 ((with-validated-input (s (read-line) stringp) (length s))
  . (LET ((S (READ-LINE)))
      (UNLESS (STRINGP S) (ERROR "invalid input: ~A" S))
      (LENGTH S)))

 ((with-validated-input (port (parse-port arg) valid-port-p) (start-server port))
  . (LET ((PORT (PARSE-PORT ARG)))
      (UNLESS (VALID-PORT-P PORT) (ERROR "invalid input: ~A" PORT))
      (START-SERVER PORT)))
)
