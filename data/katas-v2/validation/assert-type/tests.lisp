;;; tests.lisp — hand-crafted, CL idiom
;;; instruction: Write `assert-type` — signal TYPE-ERROR if a place's
;;; value doesn't satisfy the named type.
;;; category: validation | technique: type-check | complexity: basic
;;; quality_score: 0.95
;;; Note: expected expansions show the post-unwind form (UNLESS → IF).

(
 ((assert-type x integer)
  . (IF (TYPEP X 'INTEGER) NIL
        (ERROR 'TYPE-ERROR :DATUM X :EXPECTED-TYPE 'INTEGER)))

 ((assert-type port (integer 1 65535))
  . (IF (TYPEP PORT '(INTEGER 1 65535)) NIL
        (ERROR 'TYPE-ERROR :DATUM PORT :EXPECTED-TYPE '(INTEGER 1 65535))))

 ((assert-type s string)
  . (IF (TYPEP S 'STRING) NIL
        (ERROR 'TYPE-ERROR :DATUM S :EXPECTED-TYPE 'STRING)))
)
