
%% given a bhv2 file plots overview across trials
clear all;
close all;

%% PARAMS
animal = 'otis';
datethis = '200106';

%%
datdir = ['/data2/animals/Otis/' datethis];
datfile = '200106_131113_pilot_Otis_1.bhv2';
datfile = '200106_133911_pilot_Otis_7.bhv2';

datdir = ['/data2/animals/Mo/' datethis];
datfile = '200106_122603_pilot_Mo_11.bhv2';

[data,MLConfig,TrialRecord,filename,varlist] = mlread([datdir '/' datfile]);
% [data,MLConfig,TrialRecord,filename,varlist] = mlread();

% PIX_PER_DEG = 26.646;
PIX_PER_DEG  = MLConfig.PixelsPerDegree(1);
assert(MLConfig.PixelsPerDegree(2) == -MLConfig.PixelsPerDegree(1))

% PIX_PER_UNIT = [1024 768];


%%

H = str2num(MLConfig.Resolution(1:strfind(MLConfig.Resolution, ' x')-1));

ind1 = strfind(MLConfig.Resolution, 'x ')+1;
ind2 = strfind(MLConfig.Resolution, ' 59 Hz')-1;
W = str2num(MLConfig.Resolution(ind1:ind2));

PIX_PER_UNIT = [H W];


FS = MLConfig.AISampleRate;

%%


for i=1:20
% i = 6;
COLOR_EACH_STROKE = 1;


% ntrials = min([length(data), 5]);
% triallist = find([data.TrialError]==0);

% metadat
tnum = data(i).Trial;
errorcode = data(i).TrialError;

% touch data
dat = data(i).AnalogData.Touch;
x_touch = dat(:,2).*PIX_PER_DEG;
y_touch = -dat(:,1).*PIX_PER_DEG;

% dat = dat.*PIX_PER_DEG;
% x_touch = dat(:,1);
% y_touch = dat(:,2);

% only keep times when touching.
indsnotnan = find(~isnan(dat(:,1)));
% segment into onsets and offsets
onsets = find(diff(isnan(dat(:,1)))==-1)+1;
if ~isnan(dat(1,1))
onsets = [1; onsets];
end

offsets = find(diff(isnan(dat(:,1)))==1);
offsets = [offsets; length(dat(:,1))];

tvals = (1:size(dat,1))/FS; % convert to sec.
% dat = dat(indsnotnan, :);


% PLOT
figure;

% 1) Plot touch data for entire trial.
subplot(2,2,1); hold on;
title(['tnum ' num2str(tnum) ', trialerror ' num2str(errorcode)]);
%     plot(dat(:,1), dat(:,2), 'ok');
scatter(x_touch, y_touch, [], tvals, 'o');
colormap('spring')

xlim([-1024/2 1024/2]);
ylim([-768/2 768/2]);

% 2) Plot timecourse over trial.
subplot(2,2,2); hold on;
frac_ink_gotten = sum(TrialRecord.User.InkGotten{i})/length(TrialRecord.User.InkGotten{i});
title(['ink gotten : ' num2str(100*frac_ink_gotten) '%']);
plot(tvals, x_touch, 'or');
plot(tvals, y_touch, 'ob');
ylabel('x(red), y(blue)');
xlabel('time (sec)');

% overlay strokes segmented
yy = max([x_touch; y_touch])+5;
for k=1:length(onsets)
    try
    line([tvals(onsets(k)) tvals(offsets(k))], [yy yy], 'Color', 'k', 'LineWidth', 3);
    catch err
    end
end

% Overlay times of behavioral codes.
datcodes = data(i).BehavioralCodes;
for k=1:length(datcodes.CodeTimes)
    t = datcodes.CodeTimes(k)/1000;
    c = datcodes.CodeNumbers(k);
    plot(t,yy+k+20, 'xc');
    text(t,yy+k+20, num2str(c), 'fontsize', 12);
end

stimonset = datcodes.CodeTimes(datcodes.CodeNumbers==20)/1000;
if ~isempty(stimonset)
line([stimonset stimonset], ylim);
end

% plot reward records
datrew = data(i).RewardRecord;
for k=1:length(datrew.StartTimes)
    st = datrew.StartTimes(k)/1000;
    en = datrew.EndTimes(k)/1000;
    line([st en], [yy+25+k yy+25+k], 'color', 'c', 'LineWidth', 2);
end


ylim([-max(PIX_PER_UNIT)/2 max(PIX_PER_UNIT)/2]);

% plot ground truth task
subplot(2,2,3); hold on;
task = TrialRecord.User.CurrentTask(i);
title([task.str '(stage: ' task.stage ') [during stim]']);
% plot(task.x, -task.y, 'ok');
% if (0)
%     x = TrialRecord.User.InkPositions{trial}(:,1);
%     y = TrialRecord.User.InkPositions{trial}(:,2);
% else
%     x_task = task.x;
%     y_task = task.y;
% end
% x = x*PIX_PER_UNIT(1) - PIX_PER_UNIT(1)/2;
% y = y*PIX_PER_UNIT(2) - PIX_PER_UNIT(2)/2;
% x_task = (task.x-0.5)*PIX_PER_UNIT(1);
% y_task = -(task.y-0.5)*PIX_PER_UNIT(2);
x_task = -(task.y - 0.5)*PIX_PER_UNIT(2);
y_task = -(task.x - 0.5)*PIX_PER_UNIT(1);
plot(x_task, y_task, 'xk');
ylim([-1024/2 1024/2]);
xlim([-768/2 768/2]);
lt_plot_zeroline;
lt_plot_zeroline_vert;

taskonset = datcodes.CodeTimes(datcodes.CodeNumbers==15)/1000;
taskoffset = datcodes.CodeTimes(datcodes.CodeNumbers==50)/1000;

indsthis = tvals>taskonset & tvals<taskoffset;
scatter(x_touch(indsthis), y_touch(indsthis), [], tvals(indsthis), 'o');
colormap('spring');
% xlim([0 1]);
% ylim([0 1]);
% xlim([-1024/2 1024/2]);
% ylim([-768/2 768/2]);

subplot(2,2,4); hold on;
title('color = stroke');
% pcols = lt_make_plot_colors(length(onsets), 1, [1 0 0]);
pcols = cool(length(onsets));
for k=1:length(onsets)
    try
    indsthis = onsets(k):offsets(k);
   plot(x_touch(indsthis), y_touch(indsthis), '-o', 'Color', pcols(k, :), 'LineWidth', 2);
%    plot(x_touch(indsthis), y_touch(indsthis), '-', 'Color', pcols(k, :), 'LineWidth', 2);
   
    catch err
    end
end
plot(x_task, y_task, 'xk');
ylim([-1024/2 1024/2]);
xlim([-768/2 768/2]);
lt_plot_zeroline;
lt_plot_zeroline_vert;



end