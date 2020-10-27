function W_lda = MvDAvc(X_multiview_path,Label_multiview_path,lambda)
load(X_multiview_path);
load(Label_multiview_path);
num_class = length(unique(Label_multiview{1}));
num_view = size(X_multiview,2);

% dim = size(X_multiview{1},1);
%% ************** Sjr ********************************************
A = cell(num_view, 3);
for vi = 1:num_view
    [Xi ci li] = GetEachClass(X_multiview{vi},Label_multiview{vi},'x');
    [Mi ci li] = GetEachClass(X_multiview{vi},Label_multiview{vi},'m');
    [Numi ci li] = GetEachClass(X_multiview{vi},Label_multiview{vi},'num');
    A{vi,1} = Xi; % class data [1x700]
    A{vi,2} = Mi; % class mean [100x700]
    A{vi,3} = Numi; % class count [1x700]
end

Ni = zeros(1,num_class); % total class count [1x700]
dim = zeros(1,num_view); % view dimension (100, 100)
Pmv = cell(1,num_view); % something related to vc [700x100]

for i=1:num_view
    Ni = Ni + A{vi,3};
    dim(i) = size(X_multiview{i},1);
    Xi = X_multiview{i};
    Ireg = eye(size(Xi,2))*trace(Xi'*Xi)*0.0001;
    Pi = (Xi'*Xi+Ireg)\(Xi');
    Pmv{i} = Pi;
end

%% *********** LDA SW *******************************************
Sw = zeros(sum(dim),sum(dim));
for i=1:num_view
    Numi = A{i,3};
    Mi  = A{i,2};
    for j=i:num_view
        Numj = A{j,3};
        Mj = A{j,2};
        Xj = A{j,1};
        sij = zeros(dim(i),dim(j));
        vij = zeros(dim(j),dim(j));
        for ci = 1:num_class
            sij = sij - (Numi(ci)*Numj(ci)/Ni(ci))*(Mi(:,ci)*Mj(:,ci)');
            vij = vij + Xj{ci} * (Xj{ci}');
        end
        if j==i
            sij = vij + sij;
        end
        
        Sw(sum(dim(1:i-1))+1:sum(dim(1:i)), sum(dim(1:j-1))+1:sum(dim(1:j))) = sij;
        Sw(sum(dim(1:j-1))+1:sum(dim(1:j)), sum(dim(1:i-1))+1:sum(dim(1:i))) = sij';
    end
end

%% *********** LDA SB *******************************************
Sb = zeros(sum(dim),sum(dim));
n = sum(Ni);

for i=1:num_view
    mi = sum(X_multiview{i},2);
    Mi  = A{i,2};
    Numi = A{i,3};
    for j=i:num_view
        Numj = A{j,3};
        Mj = A{j,2};
        sij = zeros(dim(i),dim(j));

        mj = sum(X_multiview{j},2);
        for ci = 1:num_class            
            sij = sij + (Numi(ci)*Numj(ci)/Ni(ci))*(Mi(:,ci)*Mj(:,ci)');
        end
        sij = sij - mi*mj'/n;

        Sb(sum(dim(1:i-1))+1:sum(dim(1:i)), sum(dim(1:j-1))+1:sum(dim(1:j))) = sij;
        Sb(sum(dim(1:j-1))+1:sum(dim(1:j)), sum(dim(1:i-1))+1:sum(dim(1:i))) = sij';
    end
end


%% *********** LDA regularizer Mreg *******************************************
% Mreg = eye(sum(dim));

Mreg = zeros(sum(dim),sum(dim));
for i=1:num_view
    Pi = Pmv{i};
    for j=i:num_view
        Pj = Pmv{j};
        sij = Pi'*Pj;
        
        if j==i
             sij = (num_view-1)*sij;
        else
            sij = -1*sij;
        end
        
        Mreg(sum(dim(1:i-1))+1:sum(dim(1:i)), sum(dim(1:j-1))+1:sum(dim(1:j))) = sij;
        Mreg(sum(dim(1:j-1))+1:sum(dim(1:j)), sum(dim(1:i-1))+1:sum(dim(1:i))) = sij';
    end
end

% for i=1:num_view
%     Pi = Pmv{i};
%     for j=i:num_view
%         Pj = Pmv{j};
%         sij = Pi'*Pj;
%         
%        
%         Mreg(sum(dim(1:i-1))+1:sum(dim(1:i)), sum(dim(1:j-1))+1:sum(dim(1:j))) = sij;
%         Mreg(sum(dim(1:j-1))+1:sum(dim(1:j)), sum(dim(1:i-1))+1:sum(dim(1:i))) = sij';
%     end
% end

% Ireg = eye(sum(dim));
% Sw = Sw + lambda*(trace(Sw)/trace(Ireg)) * Ireg;
Sb = Sb.*num_view;

%% LDA
Sw = Sw + lambda*(trace(Sw)/trace(Mreg)) * Mreg;

W_LDA = LDAX_SwSb(Sw,Sb);
W_lda = cell(1,num_view);
for i=1:num_view
    W_lda{i} = (W_LDA(sum(dim(1:i-1))+1:sum(dim(1:i)), :));
end
fprintf('MvDA finished\n');
