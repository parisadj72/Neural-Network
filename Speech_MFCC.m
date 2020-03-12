% Mel-Freq Cepstral coeficient (MFCC) feature extraction
% function VecCep = MFCC(WaveFile)
%
% MFCC extraction routine, based on ETSI ES 201 108 (ETSI Front End).
%
% Input:   WaveFile  = vector containing raw audio data or the name of a wave file
% Output:  VecCep = matrix containing cepstral coefficients (rows) in time (columns)
%
% Digital Speech Processing Course
% University of Tehran, Faculty of New Sciences and Technologies (FNST)
% DSP Lab
% dsp.ut.ac.ir

% Hadi Veisi (h.veisi@ut.ac.ir)


function [VecCep] = Speech_MFCC(Audio)

S_in = Audio;
Fs = 8000;

% Parameters, define as global constants
FrameLen    = 20;            % frame length in mili sec
FrameShift  = 10;          % frame shift in mili sec.
E_thresh    = exp(-50);               % Log enery threshold
Fstart      = 100;             % Start frequency (Hz) for Mel-filtering
NumFilters  = 25;         % Mel filtering
CepLen      = 12;             % Cepstral coefficients
UseDeltaFeatures = 1; 	% if 1, calculate delta and delta2 features
lenVector   = CepLen + 1;             % As we are including log energy, vector length is number of coefficients + 1:

FrameLen  = floor(FrameLen*(Fs/1000));              % FrameLen = frame length (samples)
if rem(FrameLen,2)==1, FrameLen=FrameLen+1; end
FrameShift= floor(FrameShift*(Fs/1000));            % FrameShift = frame shift interval (samples)
LenFFT    = FrameLen;                               % Length of FFT
LenWave = length(S_in);                             % Length of input signal


HammingWin=hamming(FrameLen);                       % Calculate hamming window
HammingWin=HammingWin';
%     save ('HammingWin.tr','HammingWin');

DCTmat = DCT_Matrix (CepLen, NumFilters);           % Calculate DCT transform
MelMat = Mel_Matrix (Fs, LenFFT);              % Calculate Mel transform
S_dc = (S_in);
S_pre = PreEmphFilter(S_dc, 0.97);           % Pre-emphasis filter is applied
S_seg = Segment(S_pre,FrameLen, FrameShift,HammingWin); % This function chops the signal into frames, Hamming window of length FrameLen is applied
S_fft = fft(S_seg);                                     % Fourier transform (FFT) to compute the complex spectrum
S_phase = angle(S_fft(1:fix(end/2+1),:));               % Speech Phase
S_abs = abs(S_fft(1:fix(end/2+1),:));                   % Specrogram
VecSpec = S_abs;                                        % Spectrum befor Mel

[FreqRes numFrames]=size(S_abs);
for i = 1:numFrames
    % Energy calculation, For each frame, a log energy parameter is calculated.
    E_in = sum (S_abs(:,i).^2);
    % lnE is ln (log in Matlab) of E_in unless E_in < E_thresh:
    if E_in >= E_thresh
        lnE = log(E_in);
    else
        lnE = log(E_thresh);
    end
    
    E_fb = S_abs (:,i)' * MelMat;           % Mel filtering
    
    % Non-linear transformation (log), Apply a flooring such that no output is less than -50:
    S_fb = log(abs(E_fb));
    for p = 1:NumFilters
        if S_fb(p) < -50
            S_fb(p) = -50;
        end
    end
    
    c = DCTmat * S_fb';             % Cepstral coefficients (DCT)
    
    % Cepstrum calculation output
    % ------------------------------------------
    % c_out 1 to 12    =    cepstral coeffs 1 to 12
    % c_out 13         =    log energy measure
    c_out = zeros(CepLen+1, 1);
    c_out(1:CepLen) = c(1:CepLen);
    c_out(CepLen+1)   = lnE;
    
    VecCep(:, i) = c_out;
end % i

if (UseDeltaFeatures)                      % Compute Delta and Delta-Delta
    VecCep=Dt_Dt2(VecCep);
end
end

%% ===================================================== Segment
function Seg = Segment(signal,W,SP,Window)

% SEGMENT chops a signal to overlapping windowed segments
% A= SEGMENT(X,W,SP,WIN) returns a matrix which its columns are segmented
% and windowed frames of the input one dimentional signal, X. W is the
% number of samples per window, default value W=256. SP is the shift
% percentage, default value SP=0.4. WIN is the window that is multiplied by
% each segment and its length should be W. the default window is hamming
% window.
% 06-Sep-04
% Esfandiar Zavarehei

if nargin<3
    SP=.4*W;
end
if nargin<2
    W=256;
end
if nargin<4
    Window=hamming(W);
end
Window=Window(:); %make it a column vector

L=length(signal);
N=fix((L-W)/SP +1); %number of segments

Index=(repmat(1:W,N,1)+repmat((0:(N-1))'*SP,1,W))';
hw=repmat(Window,1,N);
Seg=signal(Index).*hw;
end

%% ===================================================== Freq to Mel
function mel = f2mel(x, lambda, mu)
%function mel = f2mel(x, lambda, mu)
%
% Implementation of ETSI Aurora standard inverse Mel function (5.3.5, eqn 5.55a)
%
% Inputs: x      = linear frequency
%         lambda = parameter (e.g. 2595)
%         mu     = parameter (e.g. 700)
%
% Output: mel    = mel-scaled frequency
%
% Jonathan Darch 12/12/03
%
% University of East Anglia, UK
% jonathan.darch@uea.ac.uk
% www.jonathandarch.co.uk

mel = lambda * log10(1 + (x/mu));
end
%% ============================================= Mel to Freq
function lem = mel2f(y, lambda, mu)
%function lem = mel2f(y, lambda, mu)
%
% Implementation of ETSI Aurora standard Mel function (5.3.5, eqn 5.55b)
%
% Inputs: y      = mel-scaled frequency
%         lambda = parameter (e.g. 2595)
%         mu     = parameter (e.g. 700)
%
% Output: lem    = linear frequency
%
% Jonathan Darch 12/12/03
%
% University of East Anglia, UK
% jonathan.darch@uea.ac.uk
% www.jonathandarch.co.uk

lem = mu * ( (10 ^ (y/lambda)) - 1);
end

%% ============================================= Mel Matrix
% Calculate bin centres for Mel-scale
function MelMat = Mel_Matrix (Fs, LenFFT)
mu        = 700;                    % Mel filtering (4.2.9)
lambda    = 2595;                   % Mel filtering (4.2.9)

cBin = zeros(1, 25+1);

% Calculate bins 1 to NumFilters:
for k = 1:(25)
    % Centre frequencies of filters calculated from Mel-function.
    Fcentre = mel2f( f2mel(100, lambda, mu) + k*( (f2mel(Fs/2, lambda, mu) - f2mel(100, lambda, mu) ) / (25 + 1)), lambda, mu );
    
    % In terms of FFT index, the central frequencies of the filters correspond to:
    cbin(k) = round( (Fcentre/Fs) * LenFFT );
end

cbin(25+1) = LenFFT / 2;

% Make filterbank basis vectors:
fbank = zeros(25, fix(LenFFT/2+1));

for k = 2:(25)
    for i = cbin(k-1):cbin(k),
        fbank( k, i ) = (i - cbin(k-1) + 1) / (cbin(k) - cbin(k-1) + 1);
    end
    for i = cbin(k)+1:cbin(k+1),
        fbank( k, i ) = 1 - (i - cbin(k)) / (cbin(k+1) - cbin(k) + 1);
    end
end

% Now make first channel
k = 1;

% Set cbin0:
cbin0 = round( (100/Fs) * LenFFT );

% Ensure that low frequency cut-off of channel 1 is not zero indexed
if cbin0 == 0
    cbin0 = 1;
end

for i = cbin0:cbin(k)
    fbank( k, i ) = (i - cbin0 + 1) / (cbin(k) - cbin0 + 1);
end

for i = cbin(k)+1:cbin(k+1)
    fbank( k, i ) = 1 - (i - cbin(k)) / (cbin(k+1) - cbin(k) + 1);
end

MelMat = fbank';
end

%% ============================================= DCT
% Calculate DCT coefficients:
function DCTmat = DCT_Matrix (CepLen, NumFilters)

DCTmat = zeros(CepLen, NumFilters);
for k = 1:NumFilters
    for i = 1:CepLen
        DCTmat(i, k) = ( cos( (i-1) * pi * (k-0.5) / NumFilters) );
    end
end
end

%% ============================================= DC
% 4.2.3 Offset compensation
% ----------------------------
% Input:   S_in (entire array)
% Output:  S_of (entire array)
% NB: We are applying this to the entire signal, not frames.
function S_of = RemoveDC (S_in)
% Cannot perform this operation on first sample, so set it to S_in(1):
S_of(1) = S_in(1);

% Remove DC offset using a notch filter:
for n = 2 : length(S_in)
    S_of(n) = S_in(n) - S_in(n-1) + ( S_of(n-1) * 0.999 );
end
end

%% ============================================= Pre-Emphasis
% Apply Pre-Emphasis Filter
function S_swp = PreEmphFilter(S_swp, preEmph)
S_swp = filter([1 -preEmph],1,S_swp);
end

%% ============================================= Undo Pre-Emphasis
% Undo Pre-Emphasis Filter
function DataWav = UndoPreEmphFilter(DataWav, PreEmph)
DataWav = filter(1,[1 -PreEmph], DataWav);
end

%% ============================================= Delta
function Out=Dt_Dt2(In)
%function Out=Dt_Dt2(In)
%
% Implementation of Delta and Delta-delta computation using regression
% based derviation
%
% Inputs: In     = Input Cepstral Vectors
%
% Output: Out    = Output vectors, delta and delta2 are attached to
% relevant static vectors
%
% Hadi Veisi (h.veisi@ut.ac.ir)

[CepLen, NumFrames]=size(In);
Out=zeros(CepLen*3, NumFrames);
Out(1:CepLen, 1:NumFrames)=In;

M1=3;
M2=3;
Tm1 = M1*(M1+1)*(2*M1+1)/3;
Tm2 = M2*(M2+1)*(2*M2+1)/3;
Tm3 = M2*(M2+1)*(2*M2+1)*(3*M2*M2+(3*M2)-1)/15;

for i=1:NumFrames
    % Delta
    if ((i-M1)<1 | (i+M1)> NumFrames)
        for j=1:CepLen
            Out(j+CepLen,i) = 0.0;
        end
    else
        for j=1:CepLen
            temp1 = 0;
            for t=-M1:M1
                temp1=temp1+t*Out(j,i+t);
            end
            Out(j+CepLen,i) = temp1/Tm1;
        end
    end % if
    
    % Delta-Delta
    if ((i-M2)<1 || (i+M2)>NumFrames)
        for j=0:CepLen
            Out(j+(2*CepLen),i) = 0.0;
        end
    else
        for j=1:CepLen
            temp1 = 0;
            temp2 = 0;
            for t=-M2:M2
                temp1 = temp1 + Out(j,i+t);
                temp2 = temp2 + t*t*Out(j,i+t);
            end
            Out(j+(2*CepLen),i) =2*(Tm2 * temp1 - (2*M2+1) * temp2) / ((Tm2*Tm2) - (2*M2+1)*Tm3);
        end
    end
end % for i
end
