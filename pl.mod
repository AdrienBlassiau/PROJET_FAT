param N;

param V{1..N};
param d{1..N};

var q{1..N,1..N};

maximize c : sum{i in 1..N,j in 1..N}(q[i,j]);
c1{j in 1..N} : sum{i in 1..N}(q[i,j]*d[i]) <= V[j];
c2{i in 1..N} : sum{j in 1..N}(q[i,j]) = 1;
c3{i in 1..N} : q[i,i] = 0;
c4{i in 1..N,j in 1..N} : 0 <= q[i,j] <= 1;

solve;

display q;

end;