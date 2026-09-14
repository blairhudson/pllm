(set-option :timeout 10000)

(set-logic QF_BV)
(declare-const x (_ BitVec 24))
(assert (not (= x ((_ sign_extend 8) ((_ extract 15 0) x)))))
(check-sat)
(get-value (x))
