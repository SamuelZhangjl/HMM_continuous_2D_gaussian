clear;clc;
close all;
format long

%% mean
mu = zeros(2,1,4);
mu(:,:,1) = [1.10 4.00]';
mu(:,:,2) = [3.20 2.80]';
mu(:,:,3) = [5.80 2.10]';
mu(:,:,4) = [4.10 1.30]';
mu_update = zeros(2,1,4);

%% Covariance matrix
sigma = zeros(2,2,4);
sigma(:,:,1) = [0.36 0; 0 0.49];
sigma(:,:,2) = [0.25 0; 0 0.64];
sigma(:,:,3) = [0.81 0; 0 0.36];
sigma(:,:,4) = [0.16 0; 0 0.09];
sigma_update = zeros(2,2,4);

%% observation sequence
obser_point = zeros(2,1,7);
obser_point(:,:,1) = [0.7 3.6]';
obser_point(:,:,2) = [2.8 2.9]';
obser_point(:,:,3) = [3.3 2.6]';
obser_point(:,:,4) = [5.1 2.2]';
obser_point(:,:,5) = [4.6 1.6]';
obser_point(:,:,6) = [4.0 1.5]';
obser_point(:,:,7) = [3.8 1.2]';

%% output probability densities b(o)
out_prob = zeros(4,7); % n * t

%% entry & exit & state-transition probabilities-
tran_prob = [0.82 0.15 0.03 0;0 0.9 0.08 0.02;0 0 0.87 0.12;0 0 0 0.96];
tran_update = zeros(4,4);
entry_prob=[0.9 0.1 0 0];
exit_prob=[0 0 0.01 0.04];

%% transition accumulators
tran_accumulators = zeros(4,4);
occup_accumulators = zeros(4,1);

%% output accumulators
mu_accumulators = zeros(2,1,4);
sigma_accumulators = zeros(2,2,4);
b_accumulators = zeros(4,1);

%% 
forward_likelihood = zeros(4,7); %forward likelihood
backward_likelihood = zeros(4,7); %backward likelihood
occup_likelihood = zeros(4,7);  %occupation likelihood

for time = 1:5
    %% draw the pdf for each state
    for stage = 1:4
        [X,Y] = meshgrid(-3:0.1:10,-3:0.1:10);
        const = 1/sqrt((2*pi)^2);
        const = const/sqrt(det(sigma(:,:,stage)));
        temp = [X(:)-mu(1,1,stage) Y(:)-mu(2,1,stage)];
        pdf = const * exp (-0.5 * diag(temp * inv(sigma(:,:,stage)) * temp'));
        figure(time);
        hold on
        pdf = reshape(pdf,size(X));   %here it may be length and add y in it
        surfc(X, Y, pdf, 'LineStyle', 'none');
        for t = 1:7
               temp = [obser_point(1,1,t)-mu(1,1,stage) obser_point(2,1,t)-mu(2,1,stage)];
               out_prob(stage,t) = const*exp(-0.5* diag(temp * inv(sigma(:,:,stage)) * temp'));
               plot3(obser_point(1,1,t),obser_point(2,1,t),2,'ro');
        end
        hold off
    end
   
    
    %% forward likelihood

    for stage = 1:4
       forward_likelihood(stage,1) = entry_prob(stage)* out_prob(stage,1);
    end

    for T = 2:7
        for j = 1:4
           temp = 0;
           for i = 1:4
               temp = temp + forward_likelihood(i,T-1)*tran_prob(i,j);
           end
           forward_likelihood(j,T) = temp * out_prob(j,T); 
        end
    end

    P_aout = 0;
    for i = 1:4
        P_aout = P_aout + forward_likelihood(i,7)*exit_prob(i);
    end


    %% backward procedure

    for stage = 1:4
       backward_likelihood(stage,7) =  exit_prob(stage);
    end

    for t = 1:6
        T = 7-t;
        for i = 1:4
            temp = 0;
            for j =1:4
               temp = temp + tran_prob(i,j)* out_prob(j,T+1)*backward_likelihood(j,T+1); 
            end
           backward_likelihood(i,T)=temp; 
        end
    end

    P_bout = 0;
    for i = 1:4
        P_bout = P_bout + entry_prob(i)*out_prob(i,1)*backward_likelihood(i,1);
    end


    %% occupation likelehoods
    temp = forward_likelihood .* backward_likelihood; 
    occup_likelihood = temp/P_aout;

    %% transition likelihoods
    temp = zeros(4,4,7);
    for t = 2:7
        for i =1:4
            for j = 1:4
                temp(i,j,t) = forward_likelihood(i,t-1) * tran_prob(i,j) * out_prob(j,t) * backward_likelihood(j,t);
            end
        end
    end

    tran_likeli = temp/P_aout;

    %% transition accumulators
    
    for t = 2:7
        tran_accumulators = tran_accumulators + tran_likeli(:,:,t);
    end

    for t = 1:7
        occup_accumulators = occup_accumulators + occup_likelihood(:,t);
    end
    
    %% output accumulators
    for i = 1:4   
        for t = 1:7
            mu_accumulators(:,1,i)= mu_accumulators(:,1,i)+ occup_likelihood(i,t).* obser_point(:,1,t);
        end
    end

    
        for i = 1:4
           scalar = 0;
           for t = 1:7
              temp =  obser_point(:,:,t) - mu(:,:,i);
              scalar = scalar + occup_likelihood(i,t)*(temp' * temp);
           end
           sigma_accumulators(1,1,i) = sigma_accumulators(1,1,i)+scalar;
           sigma_accumulators(2,2,i) = sigma_accumulators(2,2,i)+scalar;
        end
    

    for t = 1:7
        b_accumulators = b_accumulators + occup_likelihood(:,t);
    end
   

    %% update
    for i = 1:4
        tran_update(i,:) = tran_accumulators(i,:)/occup_accumulators(i);    %error!!!!!
    end

    for i = 1:4
        mu_update(1,1,i) = mu_accumulators(1,1,i)/b_accumulators(i);
        mu_update(2,1,i) = mu_accumulators(2,1,i)/b_accumulators(i);
        sigma_update(:,:,i) = sigma_accumulators(:,:,i)/b_accumulators(i);
    end
    
    %%reestimate
    disp('------------------distance from mu point distance to obervation points-----------------------');
    distance1 = 0;
    for t = 1:7
        for i = 1:4
            temp = obser_point(:,:,t) - mu(:,:,i);
            distance1 = distance1 + temp' * temp;
        end
    end
    disp(distance1);
    
    disp('----------------distance from mu_update point to obervation points-------------------');
    distance2 = 0;
    for t = 1:7
        for i = 1:4
            temp = obser_point(:,:,t) - mu_update(:,:,i);
            distance2 = distance2 + temp' * temp;
        end
    end
    disp(distance2); 
    
    
    tran_prob = tran_update;
    mu = mu_update;
    sigma = sigma_update;
end


