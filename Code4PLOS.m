%Detection of TBI Status using HMM
clc
clear all
rand('seed',1);
FPRstp=[0:0.001:1]; % the vector of FPR values where final ROC curve will be calculated for
K=20;% doing a k-fold cross validation for training on the subjects with good or death outcome,
WordLength=3; %The number of hours in a word

load Data
FinalDataSummary=Data{1,1}; % a matrix whose rows determine patient ID number, the timing of the begining of the  monitoring segment, the duration of the first monitoring segment,
% and total hours of available data monitoring, and total number of data
% monitoring segment for that specific patients, Age, GOS,GCS, and gender

FinalData=Data{1,2}; % a cell array where each element of the cell is a matrix of the hourly collected values of all variables for a specific patient

IGOS1=find(FinalDataSummary(:,7)==1);  %index of the  patients with GOS 1
IGOS3=find(FinalDataSummary(:,7)==3);  %index of the  patients with GOS 3
IGOS4=find(FinalDataSummary(:,7)==4);  %index of the  patients with GOS 4
IGOS5=find(FinalDataSummary(:,7)==5);  %index of the  patients with GOS 5


IndGOS1 = crossvalind('Kfold', length(IGOS1), K);  %cross validation indices for the subjects with GOS1
IndGOS3 = crossvalind('Kfold', length(IGOS3), K);
IndGOS4 = crossvalind('Kfold', length(IGOS4), K);
IndGOS5 = crossvalind('Kfold', length(IGOS5), K);

HMMResult=cell(length(FinalDataSummary),1);  % a cell represneting the detected status for all the subjects
HMMResultData=cell(length(FinalDataSummary),1);  %a cell representing the values of the all variables that was used for Detection
HMMparamAverage=cell(length(FinalDataSummary),1); %a cell representing the estimated HMM parameter values used for Detection of the status of the specific patient
% each element of  the cell is a 4 x 3 matrix, columns represent states,
% rows represent the variables

for m=1:K
    TestDPatIDGOS1=IGOS1(find(IndGOS1==m));  %Patient ID of the subjects with GOS 1 included in the testing dataset
    TestDPatIDGOS3=IGOS3(find(IndGOS3==m));
    TestDPatIDGOS4=IGOS4(find(IndGOS4==m));
    TestDPatIDGOS5=IGOS5(find(IndGOS5==m));
    
    TestDPatID=sort([TestDPatIDGOS1;TestDPatIDGOS3;TestDPatIDGOS4;TestDPatIDGOS5]); % note that these indices are in the range of 1 to length(FinalData)
    TrainDPatID=sort(setdiff([IGOS1;IGOS5],[TestDPatIDGOS1;TestDPatIDGOS5])); % note we just use subjects with GOS 1 and 5 for the training
    
    %remove the NaN from the data and put it in the correct format for HMM (inside
    %each cell, there is a matrix whose rows correspond to different features)
    
    kTrain=1;
    kTest=1;
    dataTestSummary=[];
    deathData=[]; %a matrix whose columns are ICP, CPP, RAP, PRx for the subjects with death outcome
    GoodData=[]; %a matrix whose columns are ICP, CPP, RAP, PRx for the subjects with good outcome
    for i=1:length(FinalData)
        dummy=FinalData{i,1};
        Cleandummy=dummy(~isnan(dummy(:,1))& ~isnan(dummy(:,2))&~isnan(dummy(:,5))&~isnan(dummy(:,6)),[1,2,5,6]);
        if sum(ismember(TrainDPatID,i))==1
            dataTrain{kTrain,1}=transpose(Cleandummy);
            kTrain=kTrain+1;
            if sum(ismember(IGOS1,i))==1
                dummydeath=FinalData{IGOS1,1};  %data from a subject with death outcome
                Cleandummydeath=dummydeath(~isnan(dummydeath(:,1))& ~isnan(dummydeath(:,2))&~isnan(dummydeath(:,5))&~isnan(dummydeath(:,6)),[1,2,5,6]);
                deathData=[deathData;Cleandummydeath];
            else
                dummygood=FinalData{IGOS5,1};  %data from a subject with good outcome
                Cleandummygood=dummygood(~isnan(dummygood(:,1))& ~isnan(dummygood(:,2))&~isnan(dummygood(:,5))&~isnan(dummygood(:,6)),[1,2,5,6]);
                GoodData=[GoodData;Cleandummygood];
            end
        elseif sum(ismember(TestDPatID,i))==1
            dataTest{kTest,1}=transpose(Cleandummy);
            dataTestSummary(kTest,:)=FinalDataSummary(i,:);
            kTest=kTest+1;
        end
    end
    deathCov=cov(deathData); %covariance of subjects with death data
    GoodCov=cov(GoodData); %covariance of subjects with good data
    
    %train HMM on patients with GOS of 1 and 5
    
    %initial values of the parameters
    prior0=[0.4 0.3 0.3]; %initial probability
    
    Q=3; % nr of states
    M=1; %nr of mixture of Gaussians
    O=4; %dimensionality
 
    transmat0=[0.6 0.3 0.1;0.3 0.4 0.3; 0.1 0.3 0.6];
    
    mu0(1,1:3,1)=[10 20 30]; %meanICP for all the states
    mu0(2,1:3,1)=[70 60 50]; %meanCPP for all the states
    mu0(3,1:3,1)=[0.8 0.6 0.4];%meanRAP for all the states
    mu0(4,1:3,1)=[-0.5 0 0.5];%meanPRx for all the states
    
    sigma0(1:4,1:4,1,1)=GoodCov; %covariance matrix of good outcome
    sigma0(1:4,1:4,2,1)=0.5*(deathCov+GoodCov);  %covariance matrix of gray state 1
    sigma0(1:4,1:4,3,1)=deathCov;   %covariance matrix of death outcom
    
    % we have only 1 cluster
    mixmat0 = ones(Q,1);
    [LL, prior, transmat, mu, sigma, mixmat] = ...
        mhmm_em(dataTrain, prior0, transmat0, mu0, sigma0, mixmat0, 'max_iter', 5);
    %dataTrain is a cell whose rows represent the observations (extracted features) for the corresponding
    %file
   
    % test HMM on remaining patients
    for i=1:length(TestDPatID)
        B = mixgauss_prob(dataTest{i,1}, mu, sigma);
        [path] = viterbi_path(prior, transmat, B);
        HMMResult{TestDPatID(i),1}=path-1; %detected status is a number between 0 and 2
        HMMResultData{TestDPatID(i),1}=dataTest{i,1}; %data used to predict the status
        HMMparamAverage{TestDPatID(i),1}=mu; %estimated average values of the HMM parameters used for decision making of this patient
    end
end;

%% testing the hypotheses over the data of all subjects:
%checking the mean  values of all variables between the three statuses

State0=[]; %a metrix whose columns are ICP,CPP, RAP and PRx for all the State0 statuses
State1=[];
State2=[];

for i=1: length(HMMResult)
    ICP=HMMResultData{i,1}(1,:);
    CPP=HMMResultData{i,1}(2,:);
    RAP=HMMResultData{i,1}(3,:);
    PRx=HMMResultData{i,1}(4,:);
    State0=[State0;[ICP(HMMResult{i,1}(1,:)==0)',CPP(HMMResult{i,1}(1,:)==0)',RAP(HMMResult{i,1}(1,:)==0)',PRx(HMMResult{i,1}(1,:)==0)']];
    State1=[State1;[ICP(HMMResult{i,1}(1,:)==1)',CPP(HMMResult{i,1}(1,:)==1)',RAP(HMMResult{i,1}(1,:)==1)',PRx(HMMResult{i,1}(1,:)==1)']];
    State2=[State2;[ICP(HMMResult{i,1}(1,:)==2)',CPP(HMMResult{i,1}(1,:)==2)',RAP(HMMResult{i,1}(1,:)==2)',PRx(HMMResult{i,1}(1,:)==2)']];
end
ICPdata=[State0(:,1);State1(:,1);State2(:,1)];
C=[zeros(length(State0),1);ones(length(State1),1);2*ones(length(State2),1)];
[p,~,stats]=anova1(ICPdata,C);
[c,~,~,gnames] = multcompare(stats);
ylabel('ICP values for different statuses','fontsize',20);
[p,~,stats]=anova1(ICPdata,C);
ylabel('ICP','fontsize',20);xlabel('statuses','fontsize',20);

CPPdata=[State0(:,2);State1(:,2);State2(:,2)];
C=[zeros(length(State0),1);ones(length(State1),1);2*ones(length(State2),1)];
[p,~,stats]=anova1(CPPdata,C);
[c,~,~,gnames] = multcompare(stats);
ylabel(' CPP values for different  statuses','fontsize',20);
[p,~,stats]=anova1(CPPdata,C);
ylabel('CPP','fontsize',20);xlabel('statuses','fontsize',20);

RAPdata=[State0(:,3);State1(:,3);State2(:,3)];
C=[zeros(length(State0),1);ones(length(State1),1);2*ones(length(State2),1)];
[p,~,stats]=anova1(RAPdata,C);
[c,~,~,gnames] = multcompare(stats);
ylabel('RAP values for different statuses','fontsize',20);
[p,~,stats]=anova1(RAPdata,C);
ylabel('RAP','fontsize',20);xlabel('statuses','fontsize',20);


PRxdata=[State0(:,4);State1(:,4);State2(:,4)];
C=[zeros(length(State0),1);ones(length(State1),1);2*ones(length(State2),1)];
[p,~,stats]=anova1(PRxdata,C);
[c,~,~,gnames] = multcompare(stats);
ylabel('PRx values for different statuses','fontsize',20);
[p,~,stats]=anova1(PRxdata,C);
ylabel('PRx','fontsize',20);xlabel('statuses','fontsize',20)


%checking that the percentage of time each patient spent within each state is
%significantly different between the 4 GOS

for i=1: length(HMMResult)
    State0per(i,1)=length(find(HMMResult{i,1}==0))/length(HMMResult{i,1});
    State1per(i,1)=length(find(HMMResult{i,1}==1))/length(HMMResult{i,1});
    State2per(i,1)=length(find(HMMResult{i,1}==2))/length(HMMResult{i,1});
end

GOS=FinalDataSummary(:,7);

State0Ratio=[State0per(GOS==1);State0per(GOS==2);State0per(GOS==3);State0per(GOS==4);State0per(GOS==5)];
C=[ones(length(find(GOS==1)),1);2*ones(length(find(GOS==2)),1);3*ones(length(find(GOS==3)),1);4*ones(length(find(GOS==4)),1);5*ones(length(find(GOS==5)),1)];
[p,~,stats]=anova1(State0Ratio,C);
[c,~,~,gnames] = multcompare(stats);
ylabel('GOS','fontsize',20);
title('Percentage of State0','fontsize',20);
[p,~,stats]=anova1(State0Ratio,C);
ylabel('Percentage of State0','fontsize',20);xlabel('GOS','fontsize',20);


State1Ratio=[State1per(GOS==1);State1per(GOS==2);State1per(GOS==3);State1per(GOS==4);State1per(GOS==5)];
C=[ones(length(find(GOS==1)),1);2*ones(length(find(GOS==2)),1);3*ones(length(find(GOS==3)),1);4*ones(length(find(GOS==4)),1);5*ones(length(find(GOS==5)),1)];
[p,~,stats]=anova1(State1Ratio,C);
[c,~,~,gnames] = multcompare(stats);
ylabel('GOS','fontsize',20);
title('Percentage of State1','fontsize',20);
[p,~,stats]=anova1(State1Ratio,C);
ylabel('Percentage of State1','fontsize',20);xlabel('GOS','fontsize',20);


State2Ratio=[State2per(GOS==1);State2per(GOS==2);State2per(GOS==3);State2per(GOS==4);State2per(GOS==5)];
C=[ones(length(find(GOS==1)),1);2*ones(length(find(GOS==2)),1);3*ones(length(find(GOS==3)),1);4*ones(length(find(GOS==4)),1);5*ones(length(find(GOS==5)),1)];
[p,~,stats]=anova1(State2Ratio,C);
[c,~,~,gnames] = multcompare(stats);
ylabel('GOS','fontsize',20);
title('Percentage of State2','fontsize',20);
[p,~,stats]=anova1(State2Ratio,C);
ylabel('Percentage of State2','fontsize',20);xlabel('GOS','fontsize',20);

%%
%Making a normalized histogram (pdf) of the ternary words for each of the 4
%GOS values (1,3,4,5)
PDF=zeros(4,3^WordLength); % a matrix whose rows represent the histogram of the ternary words for the patients with the corresponding GOS values
for i=1:length(HMMResult)
    Seq=HMMResult{i,1};
    PatientGOS=FinalDataSummary(i,7); % GOS value for the corresponding patient
    for j=1:length(Seq)-WordLength+1
        TerWord=sum(3.^[0:WordLength-1].*(Seq(1,j:j+WordLength-1))); % decimal number corresponding to the ternary word
        if PatientGOS==1
            PDF(PatientGOS,TerWord+1)=PDF(PatientGOS,TerWord+1)+1;
        else
            PDF(PatientGOS-1,TerWord+1)=PDF(PatientGOS-1,TerWord+1)+1;
        end 
    end
end
for i=1:4
    PDF(i,:)=PDF(i,:)/sum(PDF(i,:));
end

figure;
h=bar([0:3^WordLength-1],PDF');set(h,'linewidth',2);legend('GOS=1','GOS=3','GOS=4','GOS=5');ylim([0 1]);
title('Ternary Word Distributions', 'fontsize',20);

%% Making a normalized histogram (pdf) of the ternary words for each patient
PDFPat=zeros(length(HMMResult),3^WordLength); % a matrix whose rows represent the histogram of the ternary words for the corresponding patient
for i=1:length(HMMResult)
    Seq=HMMResult{i,1};
    for j=1:length(Seq)-WordLength+1
        TerWord=sum(3.^[0:WordLength-1].*(Seq(1,j:j+WordLength-1))); % decimal number corresponding to the ternary word
        PDFPat(i,TerWord+1)=PDFPat(i,TerWord+1)+1;
    end
    PDFPat(i,:)=PDFPat(i,:)/sum(PDFPat(i,:));
end
D1 = distEmd( PDFPat, PDFPat ); %Earth mover distance between the PDFPats of all the patients

%%Phylogenetic tree

FinalDis=zeros(4,4); 
for i=1:4
    if i==1
        PIDi=find(FinalDataSummary(:,7)==i); %PID of the patients with GOS value of i
    else
        PIDi=find(FinalDataSummary(:,7)==i+1); %PID of the patients with GOS value of i
    end
    for j=1:4
        clear DistanceMat DistanceVec
        if j==1
            PIDj=find(FinalDataSummary(:,7)==j); %PID of the patients with GOS value of j
        else
            PIDj=find(FinalDataSummary(:,7)==j+1); %PID of the patients with GOS value of j
        end
        DistanceMat=D1(PIDi,PIDj); %focusing on the distance between group i and j
        if i==j
            DistanceVec= DistanceMat(tril(logical(ones(length(DistanceMat))),-1)); %take the lower triangular elements
        else
            DistanceVec= DistanceMat(:); %add the diagonal elements
        end
        FinalDis(i,j)=mean(DistanceVec); %getting the average of the distances
    end
end
tree = seqlinkage(FinalDis);
view(tree);


%% Regression Classification using State0per and State2per 

% regression on percnetage
Data=[State0per,State2per];
Class=GOS;
Class(GOS~=1,end)=0; % detecting survival versus death
c = cvpartition(Class,'kfold',K);  %training(c,j) and test(c,j) returns the index of training and testing data points for the jth fold
ScoresRegressPer=zeros(size(Class));

for i=1:K
    X_Train=Data(training(c,i),:);
    Y_Train=Class(training(c,i),:);
    X_Test=Data(test(c,i),:);

    LabelsRegressPer{i}=Class(test(c,i));
    B=glmfit(X_Train,Y_Train,'binomial','link','logit');
    detected = glmval(B,X_Test,'logit');
    ScoresRegressPer(test(c,i))=detected;
end;
[XRegressPer,YRegressPer,TRegressPer,AUCRegressPer] = perfcurve(Class,ScoresRegressPer,'1','NBoot',500,'BootType','norm','TVals' ,[1:-0.01:0] );

%% Regression Classification using the averages of the variables
%%% Calculate the average of each variable per patient, Do regression using
% K-fold cross validation, calculate score, then do the AUC with
% bootstapping

Feat=zeros(length(FinalData),4); %A matrix whose columns are average of ICP, CPP, RAP and PRX for the specific patient
Class=FinalDataSummary(:,7);
Class(GOS~=1,end)=0; % detecting survival versus death
for i=1:length(FinalData)
    dummy=FinalData{i,1};
    Cleandummy=dummy(~isnan(dummy(:,1))& ~isnan(dummy(:,2))&~isnan(dummy(:,5))&~isnan(dummy(:,6)),[1,2,5,6]);
    Feat(i,:)=mean(Cleandummy);
end

ScoresRegressAve=zeros(size(Class));
for i=1:K
    X_Train=Feat(training(c,i),:);
    Y_Train=Class(training(c,i),:);
    X_Test=Feat(test(c,i),:);
    B=glmfit(X_Train,Y_Train,'binomial','link','logit');
    detected = glmval(B,X_Test,'logit');  %detected is the vector of detected values ( a number less than 1)
    ScoresRegressAve(test(c,i))=detected;
end
[XRegressAve,YRegressAve,TRegressAve,AUCRegressAve] = perfcurve(Class,ScoresRegressAve,'1','NBoot',500,'BootType','norm','TVals' ,[1:-0.01:0] );

% Brier Score
%Regression on percentage
BsRegressPer=(1/length(Class))*sum((ScoresRegressPer-Class).^2)
%Regression on average
BsRegressAve=(1/length(Class))*sum((ScoresRegressAve-Class).^2)


figure;
errorbar(XRegressAve(:,1),YRegressAve(:,1),YRegressAve(:,1)-YRegressAve(:,2),YRegressAve(:,3)-YRegressAve(:,1));
hold on;
errorbar(XRegressPer(:,1),YRegressPer(:,1),YRegressPer(:,1)-YRegressPer(:,2),YRegressPer(:,3)-YRegressPer(:,1),'r');
title(['AUC_(_A_v_e_r_a_g_e_s_) =',num2str(AUCRegressAve(1)),' \pm 0.07',    '     , AUC_(_P_e_r_c_e_n_a_g_e_s_) =', num2str(AUCRegressPer(1)),'\pm 0.07']);
xlim([-0.02,1.02]); ylim([-0.02,1.02]);
xlabel('False positive rate')
ylabel('True positive rate')
legend('Averages of 4 variables', 'HMM 2 percnetages');


%%Calculate Net reClassification Index on regression
for i=1:length(TRegressPer)
    pred_old=zeros(length(Class),1);
    pred_New=zeros(length(Class),1);
    pred_old(ScoresRegressAve>TRegressAve(i))=1;
    pred_New(ScoresRegressPer>TRegressAve(i))=1;
    outcome=Class;
    [ NRIRegress(i) , pvalRegress(i) ] = NetReclassificationImprovement( pred_old , pred_New , outcome );
end
[mean(NRIRegress) std(NRIRegress)]
