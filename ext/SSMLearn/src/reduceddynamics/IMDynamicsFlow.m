function [RDInfo,R,iT,N,T] = IMDynamicsFlow(etaData,varargin)
% [RDInfo,R,iT,N,T] = IMDynamicsFlow(etaData)
% Identification of the reduced dynamics in k coordinates, i.e. the vector
% field
%
%                        \dot{x} = R(x)
%
% via a weighted ridge regression. R(x) = W_r * phi(x) where phi is a
% k-variate polynomial from order 1 to order M. Cross-validation can be
% performed on random folds or on the trajectories for the map R.
% Upon request, the dynamics is returned via a coordinate change, i.e.
%
%                         R = D_T o N o iT
%
% where iT, T and N depend on the selected style.
% If the style is selected as modal, then the coordinate change is a linear
% map that transforms the linear part of R into the diagonal matrix of its
% eigenvalues.
% If the style is selected as normalform, then the functions seeks from
% data the maps iT, N and T such that the dynamics N is in normal form.
% This option is only available for purely oscillatory dynamics (i.e., the
% eigenvalues of the linear part of R are complex conjugated only). The
% normal form is detected by evaluating the small denominators that would
% occur when computing analytically the normal form from the knowledge of
% the vector field. These small denominators occur when the real part of
% the eigenvalues of the linear part of R is small or resonant. With small,
% it is intended to be smaller then a user-defined tolerance. To overcome
% tolerance-based issues, the user can enforce the detection of the
% coefficients of the normal form of the equivalent center manifold reduced
% order model, specifying eventual resonances among the frequencies.
%
% OUTPUTS
% R  - vector field in the given coordinates of etaData
% iT - transformation from the coordinates of etaData to normal form ones
% N  - vector field in the normal form coordinates
% T  - transformation from the normal form coordinates to those of etaData
% Maps_info - struct containing the information of all these mapings
%
% INPUTS
% etaData - cell array of dimension (N_traj,2) where the first column
%          contains time instances (1 x mi each) and the second column the
%          trajectories (k x mi each). Sampling time is assumed to be
%          constant
%  varargin = polynomial order of R
%     or
%  varargin = options list: 'field1', value1, 'field2', value2, ... . The
%             options fields and their default values are:
%     'c1' - error coefficient for slow manifolds weighting
%           (1+c1*exp(-c2*t)).^(-1), default 0
%     'c2' - error coefficient for slow manifolds weighting
%           (1+c1*exp(-c2*t)).^(-1), default 0
% 'l_vals' - regularizer values for the ridge regression, default 0
% 'n_folds'- number of folds for the cross validation, default 0
% 'fold_style' - either 'default' or 'traj'. The former set random folds
%               while the latter exclude 1 trajectory at time for the cross
%               validation
% 'style' - none, modal or normalform
% 'nf_style' - 'center_mfld' or 'actual eigs'
% 'tol_nf' - parameter for the tolerance in the detection of small
%            denominators in the normal form. The tolreance is set to be
%            tol_nf*max(abs(real(eig(R)))). Default value for tol_nf is 10
% 'frequencies_norm' - expected frequencies ratios (e.g. [1 2]) for 1:2
%                      resonance with the first and the second frequencies,
%                      by the default the code uses the actual values of
%                      of the frequencies. This option only works with the
%                      normal form style center manifold
% 'IC_nf' - initial condition for the optimization in the normal form.
%           0 (default): zero initial condition;
%           1: initial estimate based on the coefficients of R
%           2: normally distributed with the variance of case 1
% 'rescale' - rescale for the modal coordinates.
%           0: no rescale
%           1 (default): the maximum amplitude is 0.5 (ratios kept)
%           2: the maximum amplitude of all coordinates is 0.5
% 'fig_disp_nf' - display of the normal form.
%               0:  command line only
%               r:  LaTex-style figure with r terms per row (default 1).
%                   Command line also appears if the LaTex string is too
%                   long
%               -r: both command line and LaTex-style figure with r terms
%                   per row.
% 'fig_disp_nfp' - display of the normal form in a figure.
%               0: polar normal form display (default)
%               1: complex normal form display
%               2: both polar and complex normal form display
%              -1: no displays
% 'Display' - default 'iter'
% 'OptimalityTolerance' - default 1e-4 times the number of datapoints
% 'MaxIter' - default 1e3
% 'MaxFunctionEvaluations' - default 1e4
% 'SpecifyObjectiveGradient' - default true
%     the last five options are for Matlab function fminunc.
%     For more information, check out its documentation.

data_from_python = all(size(etaData) == [1, 2]);
if data_from_python
    disp("Data seems to have been fed from Python!")
    etaData = vertcat(etaData{:})';
end

if rem(length(varargin),2) > 0 && length(varargin) > 1
    error('Error on input arguments. Missing or extra arguments.')
end

% Reshape of trajectories into matrices
t = []; % time values
X = []; % coordinates at time k
dXdt = []; % time derivatives at time k
ind_traj = cell(1,size(etaData,1)); idx_end = 0;
for ii = 1:size(etaData,1)
    t_in = etaData{ii,1}; X_in = etaData{ii,2};
    [dXidt,Xi,ti] = finiteTimeDifference(X_in,t_in,3);
    t = [t ti]; X = [X Xi]; dXdt = [dXdt dXidt];
    ind_traj{ii} = idx_end+[1:length(ti)]; idx_end = length(t);
end
options = IMdynamics_options(nargin,varargin,ind_traj,size(X,2));
% Phase space dimension & Error Weghting
k = size(Xi,1); L2 = (1+options.c1*exp(-options.c2*t)).^(-2);
options = setfield(options,'L2',L2);

% Construct phi and ridge regression
[phi,Expmat] = multivariatePolynomial(k,1,options.R_PolyOrd);
if isempty(options.R_coeff) == 1
    if options.fig_disp_nfp ~= -1
    disp('Estimation of the reduced dynamics... ')
    end
    [W_r,l_opt,Err] = ridgeRegression(phi(X),dXdt,options.L2,...
        options.idx_folds,options.l_vals);
else
    W_r =  options.R_coeff; l_opt = 0; Err = 0;
end
R = @(x) W_r*phi(x);
R_info = assembleStruct(@(x) W_r*phi(x),W_r,phi,Expmat,l_opt,Err);
options.l = l_opt;
if options.fig_disp_nfp ~= -1
fprintf('\b Done. \n')
end
[V,D,d] = eigSorted(W_r(:,1:k));
% Find the change of coordinates desired
switch options.style
    
    case 'modal'
        % Linear transformation
        iT = @(x) V\x; T = @(y) V*y;
        T_info = assembleStruct(T,V,@(x) x,eye(k));
        iT_info = assembleStruct(iT,inv(V),@(y) y,eye(k));
        % Nonlinear modal dynamics coefficients
        V_M = multivariatePolynomialLinTransf(V,k,options.R_PolyOrd);
        W_n = V\W_r*V_M; N = @(y) V\(W_r*phi(V*y));
        N_info = assembleStruct(N,W_n,phi,Expmat);
        
    case 'normalform'
        if options.fig_disp_nfp ~= -1
        disp('Estimation of the reduced dynamics in normal form...')
        end
        n_real_eig = sum(imag(d)==0);
        if n_real_eig>0
            disp('Normal form not available. Returning modal style.')
            % Linear transformation
            iT = @(x) V\x; T = @(y) V*y;
            T_info = assembleStruct(T,V,@(x) x,eye(k));
            iT_info = assembleStruct(iT,inv(V),@(y) y,eye(k));
            % Nonlinear modal dynamics coefficients
            V_M = multivariatePolynomialLinTransf(V,k,options.R_PolyOrd);
            W_n = V\W_r*V_M; N = @(y) V\(W_r*phi(V*y));
            N_info = assembleStruct(N,W_n,phi,Expmat);
        else
            if options.rescale == 1
                v_rescale = max(abs(V\X),[],2);
                V = 2*V*diag(max(v_rescale(1:k/2))*ones(1,k));
            end
            if options.rescale == 2
                v_rescale = max(abs(V\X),[],2);
                V = 2*V*diag(v_rescale);
            end
            % Initialize the normal form
            Maps_info_opt=initialize_nf_flow(V,D,d,W_r,etaData,options);
            % Get normal form mappings T, N and T^{-1}
            Maps = dynamicsCoordChangeNF(Maps_info_opt,options);
            % Final output
            T_info_opt = Maps.T; N_info_opt = Maps.N;
            iT_info_opt = Maps.iT;
            iT  = iT_info_opt.Map; N = N_info_opt.Map; T = T_info_opt.Map;
            iT_info = assembleStruct(iT,iT_info_opt.coeff,...
                iT_info_opt.phi,iT_info_opt.Exponents);
            N_info = assembleStruct(N,N_info_opt.coeff,N_info_opt.phi,...
                N_info_opt.Exponents);
            T_info = assembleStruct(T,T_info_opt.coeff,T_info_opt.phi,...
                T_info_opt.Exponents);
            iT_info.lintransf = inv(V); T_info.lintransf = V;
            
            % Display the obtained normal form
            flag_long = 0;
            if abs(options.fig_disp_nf)>0
                [str_eqn,str_eqn_plot,flag_long] = dispNormalFormFigure(N_info_opt.coeff,...
                    N_info_opt.Exponents,abs(options.fig_disp_nf));
                N_info.LaTeXComplex = str_eqn;
                if length(str_eqn_plot)>1 && options.fig_disp_nfp ~= 0
                    figure;
                    h = plot(0,0);
                    set(gcf,'color','w');
                    str_above = ['Using the notation $\bar{\,}$ for the complex conjugates, the identified normal form is'];
                    annotation('textbox','FontSize',18,'Interpreter','latex','FaceAlpha','1','EdgeColor','w','Position',[0.01 0.1 0.99 0.9], 'String',str_above);
                    annotation('textbox','FontSize',18,'Interpreter','latex','FaceAlpha','1','EdgeColor','w','Position',[0.02 0.12 0.98 0.76],'String',['$' str_eqn_plot '$']);
                    delete(h);
                    set(gca,'Visible','off')
                end
            end
            if options.fig_disp_nfp > 0
                if (options.fig_disp_nf<=0) || (flag_long==1)
                    fprintf('\n')
                    disp(['The data-driven normal form dynamics reads:'])
                    fprintf('\n')
                    table_nf = dispNormalForm(N_info_opt.coeff,...
                        N_info_opt.Exponents);
                    disp(table_nf)
                    if k == 2
                        disp(['Notation: z is a complex number; z` is the ' ...
                            'complex conjugated of z; z^k is the k-th power of z.'])
                    else
                        disp(['Notation: each z_j is a complex number; z`_j is the '...
                            'complex conjugated of z_j; z^k_j is the k-th power of z_j.'])
                    end
                end
            end
            % Polar Normal Form
            if abs(options.fig_disp_nfp) ~= 1
                N_info = polarNormalForm(N_info,1);
            else
                N_info = polarNormalForm(N_info,0);
            end
        end
        
    otherwise
        T = @(x) x; iT=@(y) y; N =@(y) R(y);
        T_info = assembleStruct(@(x) x,eye(k),@(x) x,eye(k));
        iT_info = T_info; N_info = R_info;
end
RDInfo = struct('reducedDynamics',R_info,'inverseTransformation',...
             iT_info,'conjugateDynamics',N_info,'transformation',T_info,...
             'conjugacyStyle',options.style,'dynamicsType','flow',...
             'eigenvaluesLinPartFlow',d,'eigenvectorsLinPart',V);
end

%---------------------------Subfunctions-----------------------------------

function str_out = assembleStruct(fun,W,phi,Emat,varargin)
PolyOrder = sum(Emat(end,:));
if isempty(varargin) == 0
    str_out = struct('map',fun,'coefficients',W,'polynomialOrder',...
    PolyOrder,'phi',phi,'exponents',Emat,'l_opt',varargin{1},...
    'CV_error',varargin{2});
else
   str_out = struct('map',fun,'coefficients',W,'polynomialOrder',...
                    PolyOrder,'phi',phi,'exponents',Emat);    
end
end

%- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

function [Maps_info_opt]=initialize_nf_flow(V,D,d,W_r,etaData,options)
% Preparation function for the estimate of the normal form maps. Based on
% the optimization properties, the functions seeks the coefficients of the
% normal form dynamics and sets to zero those coefficients for the
% transformation T^{-1}. Their indexes are stored in the output struct.
% The error at time instant k for the successive optimization is
%
% Err_k = dYdt-D*Y+W_it_nl*Dphi_it(Y)*dYdt-W_n*phi_n(Y+W_it_nl*phi_it(Y))
%
% and this function also precomputes the difference dYdt-D*Y and the
% transformations Dphi_it(Y)*dYdt and phi_it(Y) for a more efficient
% optimization. The overall process consider complex numbers, and the
% conjugated are ignored.

% Modal transformation
k = size(W_r,1); ndof = k/2;
% Transformation of coordinates & nonlinear maps
[phi_it,Expmat_it] = multivariatePolynomial(k,2,options.iT_PolyOrd);
[~,Expmat_n] = multivariatePolynomial(k,2,options.N_PolyOrd);
% Reshape of trajectories into matrices
Y = []; % modal coordinates at time k
dYdt = []; % time derivatives at time k
Phi_iT_Y = []; % modal transformation at time k
DPhi_iT_dYdt = []; % time derivatives at time k
for ii = 1:size(etaData,1)
    t_in = etaData{ii,1}; Y_in = V\(etaData{ii,2});
    Phi_iT_Y_in = phi_it(Y_in);
    [dYDphi_i_dt,YDphi_i,~] = finiteTimeDifference([Y_in; Phi_iT_Y_in],...
        t_in,3);
    Y = [Y YDphi_i(1:k,:)]; dYdt = [dYdt dYDphi_i_dt(1:k,:)];
    Phi_iT_Y = [Phi_iT_Y YDphi_i(k+1:end,:)];
    DPhi_iT_dYdt = [DPhi_iT_dYdt dYDphi_i_dt(k+1:end,:)];
end
Y_red = Y(1:ndof,:); dYdt_red = dYdt(1:ndof,:);
d_red = diag(D); d_red=d_red(1:ndof);
V_M = multivariatePolynomialLinTransf(V,k,options.R_PolyOrd);
W_modal = V\W_r*V_M;
% Get the terms of the normal form
if strcmp(options.nf_style,'center_mfld')
    tol_nf = 1e-8;
    if isempty(options.frequencies_norm) == 1
        d_nf = 1i*imag(d);
    else
        d_nf = transpose([+1i*options.frequencies_norm ...
            -1i*options.frequencies_norm]*imag(d(1)));
    end
else
    d_nf = d; tol_nf = options.tol_nf*max(abs(real(d_nf)));
end

lidx_n = find(abs(repmat(d_nf,1,size(Expmat_n,1))-...
    repmat(transpose(Expmat_n*d_nf),k,1))<tol_nf);
if options.R_PolyOrd>options.N_PolyOrd
    W_n_0 = W_modal(:,k+[1:size(Expmat_n,1)]);
else
    W_n_0 = [W_modal(:,k+1:end) zeros(k,size(Expmat_n,1)+k-size(W_r,2))];
end
if options.R_PolyOrd>options.iT_PolyOrd
    W_it_0 = -W_modal(:,k+[1:size(Expmat_it,1)]);
else
    W_it_0 =-[W_modal(:,k+1:end) zeros(k,size(Expmat_it,1)+k-size(W_r,2))];
end
W_it_0=W_it_0./(repmat(d,1,size(Expmat_it,1))-...
    repmat(transpose(Expmat_it*d),k,1));
lidx_elim_it = lidx_n;
if options.iT_PolyOrd<options.N_PolyOrd
    lidx_elim_it(lidx_elim_it>numel(W_it_0)) = [];
end
lidx_it  = transpose(1:numel(W_it_0));
lidx_it(lidx_elim_it)  = [];
% Set the indexes for the coefficients of T^{-1} and N
[idx_it(:,1),idx_it(:,2)] = ind2sub(size(W_it_0),lidx_it);
idx_it(idx_it(:,1)>ndof,:) = []; % Eliminate cc rows
W_it_0_up = W_it_0(1:ndof,:);
lidx_it_up = sub2ind(size(W_it_0_up),idx_it(:,1),idx_it(:,2));
% Eliminate uncessary exponents for N
[idx_n(:,1),  idx_n(:,2)] = ind2sub(size(W_n_0),lidx_n);
idx_n(idx_n(:,1)>ndof,:) = []; % Eliminate cc rows
W_n_0_up = W_n_0(1:ndof,:);
lidx_n_up = sub2ind(size(W_n_0_up),idx_n(:,1),idx_n(:,2));
W_n_0_up(setdiff(1:numel(W_n_0_up),lidx_n_up)) = 0;
IDX_expnts = ones(size(W_n_0_up));
IDX_expnts(setdiff(1:numel(W_n_0_up),lidx_n_up)) = 0;
idx_expnts = find(sum(IDX_expnts,1));
[phi_n,Expmat_n,D_phi_n_info] = multivariatePolynomialSelection(k,2,...
    options.N_PolyOrd,idx_expnts);
W_n_0_up = W_n_0_up(:,idx_expnts); IDX_expnts = IDX_expnts(:,idx_expnts);
lidx_n_up = find(IDX_expnts);
[idx_n(:,1),  idx_n(:,2)] = ind2sub(size(W_n_0_up),lidx_n_up);
% Initial condition for the optimization
if ndof == 1
    IC_opt_complex =transpose([W_it_0_up(lidx_it_up) W_n_0_up(lidx_n_up)]);
else
    IC_opt_complex = [W_it_0_up(lidx_it_up); W_n_0_up(lidx_n_up)];
end
IC_opt = [real(IC_opt_complex); imag(IC_opt_complex)];
% Maps info
N_info_opt  = struct('phi',phi_n,'Exponents',Expmat_n,'idx',idx_n,...
    'lidx',lidx_n_up);
iT_info_opt = struct('phi',phi_it,'Exponents',Expmat_it,'idx',idx_it,...
    'lidx',lidx_it_up);
D_phi_n = cell2struct(D_phi_n_info,{'Derivative','Indexes'},2);

% Final Output
Y_1_DY_red = dYdt_red - d_red.*Y_red;
Maps_info_opt = struct('IC_opt',IC_opt,'V',V,'d_r',d_red,'Yk_r',Y_red,...
    'Yk_1_r',dYdt_red,'Yk_1_DYk_r',Y_1_DY_red,...
    'Phi_iT_Yk',Phi_iT_Y,'Phi_iT_Yk_1',DPhi_iT_dYdt,...
    'iT',iT_info_opt,'N',N_info_opt,...
    'D_phi_n_info',D_phi_n);
end

%- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

% Default options

function options = IMdynamics_options(nargin_o,varargin_o,idx_traj,Ndata)
options = struct('Style','default','R_PolyOrd', 1,'iT_PolyOrd',1,...
    'N_PolyOrd',1,'T_PolyOrd',1,'c1',0,'c2',0,...
    'L2',[],'n_folds',0,'l_vals',0,'idx_folds',[],'fold_style',[],...
    'style','none','nf_style','center_mfld','frequencies_norm',[],...
    'tol_nf',1e1,'IC_nf',1,'R_coeff',[],'rescale',1,'fig_disp_nf',1,...
    'fig_disp_nfp',0,'Display','iter',...
    'OptimalityTolerance',10^(-8-floor(log10(Ndata))),...
    'MaxIter',1e3,...
    'MaxFunctionEvaluations',1e4,...
    'SpecifyObjectiveGradient',true);
% Default case
if nargin_o == 2; options.R_PolyOrd = varargin_o{:};
    options.N_PolyOrd = varargin_o{:}; end
% Custom options
if nargin_o > 2
    for ii = 1:length(varargin_o)/2
        options = setfield(options,varargin_o{2*ii-1},...
            varargin_o{2*ii});
    end
    % Some default options for polynomial degree
    if strcmp(options.style,'normalform')==1 && ...
            options.iT_PolyOrd*options.N_PolyOrd*options.T_PolyOrd == 1
        options.N_PolyOrd = options.R_PolyOrd;
    end
    if strcmp(options.style,'normalform')==1 && options.N_PolyOrd > 1
        if options.T_PolyOrd*options.T_PolyOrd == 1
            options.T_PolyOrd = options.N_PolyOrd;
            options.iT_PolyOrd = options.N_PolyOrd;
        else
            PolyOrdM = max([options.T_PolyOrd options.iT_PolyOrd]);
            options.T_PolyOrd = PolyOrdM;
            options.iT_PolyOrd = PolyOrdM;
        end
    end
    % Fold indexes
    if options.n_folds > 1
        if strcmp(options.fold_style,'traj') == 1
            options = setfield(options,'n_folds',length(idx_traj));
            idx_folds = idx_traj;
        else
            idx_folds = cell(options.n_folds,1);
            ind_perm = randperm(Ndata);
            fold_size = floor(Ndata/options.n_folds);
            for ii = 1:options.n_folds-1
                idx_folds{ii} = ind_perm(1+(ii-1)*fold_size:ii*fold_size);
            end
            ii = ii+1;
            idx_folds{ii} = ind_perm(1+(ii-1)*fold_size:length(ind_perm));
        end
        options = setfield(options,'idx_folds',idx_folds);
    end
end
end

