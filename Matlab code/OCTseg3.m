% Semi-auto segment of OCT images from 4 radial scan
% 2018.01  First Vesrion
% 2019.01.25 add non-EDI mode
% V2.0 2019.02.10  add fold info
%    add Seed function
%    add Clear .csv
%    add no good marker
% V2.1 2019.02.15  Combine R&A and Export image
% V2.2 2019.02.21 Add processing all folds for the day
%    add scale for display OCT image
% V2.2 2019.09.14 OCTseg2 modified to read Analyze image format
% V2.3 2019.11.26  add Export registed images and segments
% V3.0 combine all functions
% V3.1 add 500 pixel images
% V3.2 add average and difference for combine CSV


function OCTseg3
    oct=[];   % oct image data
    CursorckF=0;  % flag for cursor selection
    Contrast='A';  % default Auto contrast
    AllTraceckF=0;  % flag for AllTrace selection
    scl=1;  % scale factor for OCT image display

    files = {};  % file cell array
    filename = {};  % filename cell array
    pathname = [];  % pathname
%     f = 1;  % number of files
    sno=1;   % strip no
    sgLayer=2;  % default olm layer 

    segdata=[];  % segmented data
    Data=[]; % part of OCT data for segment
    fid=1;  % output file
    fidx=1;  %image index
    ftotal=1; %total image files
    review=0;  %  old segdata already exist
    old_segdata=[]; % old segdata
    
    eyefoldnames=[];  % list of fold names for OCT data
    eyeidx=1;   % eye index
    dayfold=[];   % path for day fold
    dayfoldf=0;  % flag for read more than one eye data
    
    EDIckF=1; % flag for EDI mode
    csvname=[];  % filename for .csv output
    
    AnalyzeF=0;  % not Analuze Image format
    FoLDckF=0;  % flag for folder check
    Foldpath=' ';  % folder path
    
%     ImgPxF=0;  % image pixel flag
%     ImgPixel=1024;
    

    fg = figure('NumberTitle','off');
        screensize = get( 0, 'Screensize' );
        set(fg, 'Position', [200,screensize(4)-1000,1300,900]);
        set(fg, 'Name', ['OCT Segment(v3.2)' blanks(150) 'Visual Function Core, National Eye Institute']);
    
    TopPanel = uipanel('BackgroundColor', [0.9,0.9,0.9],'Position', [0,0.9,0.85,0.1]); 
        FLoad = uicontrol('Parent', TopPanel,'Position',[880,50,100,30], 'Callback', @LoadFoldCb);
        set(FLoad, 'FontSize', 12, 'String', 'Load Folder');
        
        EDIcheck = uicontrol('Parent', TopPanel,'Position',[900,20,18,18], 'Style', 'checkbox', 'callback', @EDIcheckCb);
        set(EDIcheck, 'value', EDIckF);
        DataTxt = uicontrol('Parent', TopPanel,'Position',[915,10,40,30], 'Style', 'text');
        set(DataTxt, 'BackgroundColor', [0.9,0.9,0.9],'FontSize', 12,'ForegroundColor', 'r','String', 'EDI');

%         ImgPxcheck = uicontrol('Parent', TopPanel,'Position',[1010,50,18,18], 'Style', 'checkbox', 'callback', @ImgPxcheckCb);
%         set(ImgPxcheck, 'value', ImgPxF);
%         DataTxt = uicontrol('Parent', TopPanel,'Position',[1025,40,60,30], 'Style', 'text');
%         set(DataTxt, 'BackgroundColor', [0.9,0.9,0.9],'FontSize', 12,'ForegroundColor', 'r','String', 'ImgPx');
%         ImgPxTxt = uicontrol('Parent', TopPanel,'Position',[1010,15,60,20], 'Style', 'edit');
%         set(DataTxt, 'BackgroundColor', [0.9,0.9,0.9],'FontSize', 12,'ForegroundColor', 'r','String', 'ImgPx');
 
        uicontrol('Parent', TopPanel,'Position',[180,30,650,30], 'Style', 'Slider','SliderStep', [1/19, 1/19], ...
            'Min',1, 'Max',20, 'Value',1, 'Callback', @SlctSpCb);
        TraceSltTxt = uicontrol('Parent', TopPanel,'Position',[460,80,80,30], 'Style', 'text');
        set(TraceSltTxt, 'BackgroundColor', [0.9,0.9,0.9],'FontSize', 12,'String', 'No. of strip');
        DataTxt = uicontrol('Parent', TopPanel,'Position',[80,30,50,28], 'Style', 'text');
        set(DataTxt, 'BackgroundColor', [1,1,1],'FontSize', 14,'ForegroundColor', [0,0,0],'String', num2str(sno));
        
        AllTracecheck = uicontrol('Parent', TopPanel,'Position',[35,20,18,18], 'Style', 'checkbox', 'callback', @AllTracecheckCb);
        set(AllTracecheck, 'value', AllTraceckF);
        DataTxt = uicontrol('Parent', TopPanel,'Position',[8,35,70,30], 'Style', 'text');
        set(DataTxt, 'BackgroundColor', [0.9,0.9,0.9],'FontSize', 12,'ForegroundColor', 'k','String', 'AllTrace');
        
     CtrPanel= uipanel('BackgroundColor', [0.9,0.9,0.9],'Position', [0.85,0,0.075,0.81]); 
        RAcheck = uicontrol('Parent', CtrPanel,'Position',[15,425,60,30], 'callback', @RAcheckCb);
        set(RAcheck, 'FontSize', 14,'ForegroundColor', 'k','String', 'R&A');
        Exportcheck = uicontrol('Parent', CtrPanel,'Position',[15,345,60,30], 'callback', @ExportCb);
        set(Exportcheck, 'FontSize', 14,'ForegroundColor', 'k','String', '.tif Exp');
        
        Dayfoldcheck = uicontrol('Parent', CtrPanel,'Position',[10,255,80,30], 'callback', @DayfoldCb);
        set(Dayfoldcheck, 'FontSize', 14,'ForegroundColor', 'k','String', 'Day_fold');
        Combinecheck = uicontrol('Parent', CtrPanel,'Position',[10,165,80,30], 'callback', @CombineCb);
        set(Combinecheck, 'FontSize', 14,'ForegroundColor', 'm','String', 'Sum .csv');
        Intensitycheck = uicontrol('Parent', CtrPanel,'Position',[10,85,80,30], 'callback', @IntensityCb);
        set(Intensitycheck, 'FontSize', 14,'ForegroundColor', 'm','String', 'Intensity');
        Hypobandcheck = uicontrol('Parent', CtrPanel,'Position',[5,20,90,30], 'callback', @HypoCb);
        set(Hypobandcheck, 'FontSize', 14,'ForegroundColor', 'm','String', 'Hypoband');
       
        
%         uicontrol('Parent', CtrPanel,'Position',[40,10,30,100], 'Style', 'Slider','SliderStep', [1/9, 1/9], ...
%             'Min',1, 'Max',10, 'Value',10, 'Callback', @ScaleCb);
%         TraceSltTxt = uicontrol('Parent', CtrPanel,'Position',[20,110,80,30], 'Style', 'text');
%         set(TraceSltTxt, 'BackgroundColor', [0.9,0.9,0.9],'FontSize', 14,'String', 'Scale');
        
     
     
     LeftPanel = uipanel('BackgroundColor', [0.9,0.9,0.9],'Position', [0,0,0.075,0.9]);
        segChoice = uibuttongroup('Parent', LeftPanel,'BackgroundColor', [0.9,0.9,0.9],'Position', [0.05,0.1,0.9,0.75] , 'SelectionChangedFcn',@segChoiceCb);
            uicontrol(segChoice, 'style','radiobutton','BackgroundColor', [0.9,0.9,0.9],...
                'ForegroundColor', [1,0,1],'FontSize', 14,'String', '2, olm','Position', [10,180,100,30]);
            uicontrol(segChoice, 'style','radiobutton','BackgroundColor', [0.9,0.9,0.9],...
                'ForegroundColor', [1,0,1],'FontSize', 14,'String', '3, bm','Position', [10,5,100,30]);
            uicontrol(segChoice, 'style','radiobutton','BackgroundColor', [0.9,0.9,0.9],...
                'ForegroundColor', [1,0,1],'FontSize', 14,'String', '1, ilm','Position', [10,500,100,30]);
        DataTxt = uicontrol('Parent', LeftPanel,'Position',[0,740,90,45], 'Style', 'text');
        set(DataTxt, 'BackgroundColor', [0.9,0.9,0.9],'ForegroundColor', 'r','FontSize', 14,'String', 'Segment line');
        Seedcheck = uicontrol('Parent', LeftPanel,'Position',[15,385,60,30], 'callback', @SeedcheckCb);
        set(Seedcheck, 'FontSize', 14,'ForegroundColor', 'c','String', 'Seed');
        CSV_Clear = uicontrol('Parent', LeftPanel,'Position',[5,30,80,30], 'Callback', @CSV_ClearCb);
        set(CSV_Clear, 'FontSize', 14, 'ForegroundColor','y','String', 'Clear');
        Autocheck = uicontrol('Parent', LeftPanel,'Position',[15,155,60,30], 'callback', @AutocheckCb);
        set(Autocheck, 'FontSize', 14,'ForegroundColor', 'y','String', 'Auto');

      RightPanel = uipanel('BackgroundColor', [0.9,0.9,0.9],'Position', [0.775,0,0.075,0.9]);
         CtsChoice = uibuttongroup('Parent', RightPanel,'BackgroundColor', [0.9,0.9,0.9],'Position', [0,0.7,0.98,0.2] , 'SelectionChangedFcn',@CtsChoiceCb);
            uicontrol(CtsChoice, 'style','radiobutton','BackgroundColor', [0.9,0.9,0.9],...
                'ForegroundColor', [1,0,1],'FontSize', 14,'String', 'Auto','Position', [10,50,100,30]);
            uicontrol(CtsChoice, 'style','radiobutton','BackgroundColor', [0.9,0.9,0.9],...
                'ForegroundColor', [1,0,1],'FontSize', 14,'String', 'Super','Position', [10,5,100,30]);
            uicontrol(CtsChoice, 'style','radiobutton','BackgroundColor', [0.9,0.9,0.9],...
                'ForegroundColor', [1,0,1],'FontSize', 14,'String', 'None','Position', [10,100,100,30]);
        DataTxt = uicontrol('Parent', RightPanel,'Position',[5,750,85,30], 'Style', 'text');
        set(DataTxt, 'BackgroundColor', [0.9,0.9,0.9],'FontSize', 16,'ForegroundColor', 'r','String', 'Contrast');
        
        
        Up = uicontrol('Parent', RightPanel,'Position',[15,210,50,50], 'Callback', @UpCb);
        set(Up, 'FontSize', 12, 'String', 'Up');
        Dn = uicontrol('Parent', RightPanel,'Position',[15,150,50,50], 'Callback', @DnCb);
        set(Dn, 'FontSize', 12, 'String', 'Dn');
        
        Cursorcheck = uicontrol('Parent', RightPanel,'Position',[10,325,80,30], 'callback', @CursorcheckCb);
        set(Cursorcheck, 'FontSize', 14,'ForegroundColor', 'b','String', 'Cursor');
        
        NGcheck = uicontrol('Parent', RightPanel,'Position',[20,275,60,20], 'callback', @NGcheckCb);
        set(NGcheck, 'FontSize', 8,'ForegroundColor', 'm','String', 'No Good');
       
        Accept = uicontrol('Parent', RightPanel,'Position',[10,50,80,30], 'Callback', @AcceptCb);
        set(Accept, 'FontSize', 14, 'ForegroundColor','r','String', 'Accept');

    ThirdPanel= uipanel('BackgroundColor', [0.9,0.9,0.9],'Position', [0.925,0,0.075,0.81]); 
        Sortcheck = uicontrol('Parent', ThirdPanel,'Position',[10,435,80,30], 'callback', @SortCb);
        set(Sortcheck, 'FontSize', 14,'ForegroundColor', 'k','String', 'Sort&Comb');
    
%         CombineTiffcheck = uicontrol('Parent', ThirdPanel,'Position',[15,585,72,30], 'callback', @CombineTiffCb);
%         set(CombineTiffcheck, 'FontSize', 12,'ForegroundColor', 'm','String', 'Combine');
        Regcheck = uicontrol('Parent', ThirdPanel,'Position',[15,335,72,30], 'callback', @RegCb);
        set(Regcheck, 'FontSize', 14,'ForegroundColor', 'k','String', 'Reg w R');
        SegMeancheck = uicontrol('Parent', ThirdPanel,'Position',[15,235,72,30], 'callback', @SegMeanCb);
        set(SegMeancheck, 'FontSize', 14,'ForegroundColor', 'b','String', 'Seg Mean');
        RegExpcheck = uicontrol('Parent', ThirdPanel,'Position',[10,125,80,30], 'callback', @RegExpCb);
        set(RegExpcheck, 'FontSize', 14,'ForegroundColor', 'b','String', 'Reg Exp');

    
    InfoPanel= uipanel('BackgroundColor', [0.9,0.9,0.9],'Position', [0.85,0.81,0.15,1]); 
        InfoTxt = uicontrol('Parent', InfoPanel,'Position',[5,100,150,30], 'Style', 'text');
        set(InfoTxt, 'BackgroundColor', [0.9,0.9,0.9],'FontSize', 16,'ForegroundColor', 'r','String', 'Welcome');
        FoLD = uicontrol('Parent', InfoPanel,'Position',[20,45,100,30], 'Callback', @FoLDCb);
        set(FoLD, 'FontSize', 12, 'String', 'Folder');
        FoLDcheck = uicontrol('Parent', InfoPanel,'Position',[140,50,18,18], 'Style', 'checkbox', 'callback', @FoLDcheckCb);
        set(FoLDcheck, 'value', FoLDckF);
        FoldTxt = uicontrol('Parent', InfoPanel,'Position',[5,5,250,25], 'Style', 'text');
        set(FoldTxt, 'BackgroundColor', 'w','FontSize',10,'ForegroundColor', 'k','String', Foldpath);
        
        
    axes('outerPosition', [-0.04,-0.08,0.9,1]);
       


    function LoadFoldCb(h,e)  % load '.tif' files
        dayfoldf=0;          % not reading more than one eye
        AnalyzeF=0;       % not Analyze image format
        readfold(uigetdir)
    end

    function FoLDCb(h,e) 
        Foldpath=uigetdir;
        FoldTxt = uicontrol('Parent', InfoPanel,'Position',[5,5,250,25], 'Style', 'text');
        set(FoldTxt, 'BackgroundColor', 'w','FontSize',10,'ForegroundColor', 'k','String', Foldpath);
    end


    function DayfoldCb(h,e)  % load day fold
        dayfold = uigetdir;   % process fold list
        eyefold = dir(dayfold);
        eyeidx=1;
        dayfoldf=1;
        AnalyzeF=0;
        eyefoldnames = {eyefold.name};
        dirFlags = [eyefold.isdir] & ~strcmp(eyefoldnames, '.') & ~strcmp(eyefoldnames, '..');
        eyefoldnames = eyefoldnames(dirFlags);
        
        while isempty(dir([dayfold '\' eyefoldnames{eyeidx} '\*001.tiff']))
            eyeidx=eyeidx+1;
        end
            
        readfold([dayfold '\' eyefoldnames{eyeidx}])
    end

    function SortCb(h,e)  % Sort fold accorrding to each eye
        if FoLDckF
            dayfold=Foldpath;
        else
            dayfold = uigetdir;   % process fold list
        end
        eyefold = dir(dayfold);
%         eyeidx=1;
%         dayfoldf=1;
%         AnalyzeF=0;
        eyefoldnames = {eyefold.name};
        dirFlags = [eyefold.isdir] & ~strcmp(eyefoldnames, '.') & ~strcmp(eyefoldnames, '..');
        eyefoldnames = eyefoldnames(dirFlags);
        
        foldlist=blanks(5);
        for i=1:length(eyefoldnames)
            while isempty(dir([dayfold '\' eyefoldnames{i} '\*001.tiff']))
               i=i+1;
               if i>length(eyefoldnames); return; end
            end
            k=strfind(eyefoldnames{i}, '_');
            eyename=eyefoldnames{i}(1:k(2)-1);
            if ~contains(foldlist, eyename)
                mkdir([dayfold '\' eyename]);
                foldlist=[foldlist eyename];
            end
            movefile([dayfold '\' eyefoldnames{i}], [dayfold '\' eyename]);
        end
        
       Combine(dayfold)
       InfoDisp('Analyze Files Compiled');

    end



    function SegMeanCb(h,e)
        fold_name = uigetdir;
            files = dir([fold_name '\' '*MEAN.hdr']);   % check for averaged image in Analyze format
            pathname = [fold_name '\'];
            
            DataTxt = uicontrol('Parent', TopPanel,'Position',[100,0,470,30], 'Style', 'text');  %display fold name
            fld=strsplit(fold_name,'\');
            set(DataTxt, 'BackgroundColor', [0.9,0.9,0.9],'FontSize', 14,'ForegroundColor', 'b','String', fld{end});
            
            filename = {files.name};
            ftotal = length(filename);
            k=strfind(fold_name, '\');
            k=k(end);
            csvname=strcat(pathname, fold_name(k+1:end), '.csv');
            
            fidx=1;
            if exist(csvname,'file')
                review=1;
                old_segdata=xlsread(csvname);
            else
                review=0; 
            end
        
            fid = fopen(csvname, 'w');
            AnalyzeF=1;
            readData;
    end

    function readfold(fold_name)
            files = dir([fold_name '\' '*.tiff']);   
            pathname = [fold_name '\'];
            
            DataTxt = uicontrol('Parent', TopPanel,'Position',[100,0,470,30], 'Style', 'text');  %display fold name
            fld=strsplit(fold_name,'\');
            set(DataTxt, 'BackgroundColor', [0.9,0.9,0.9],'FontSize', 14,'ForegroundColor', 'b','String', fld{end});
            
            filename = {files.name};
            ftotal = length(filename);
            k=strfind(fold_name, '\');
            k=k(end);
            csvname=strcat(pathname, fold_name(k+1:end), '.csv');
            
            fidx=1;
            if exist(csvname,'file')
                review=1;
                old_segdata=xlsread(csvname);
            else
                review=0; 
            end
        
            fid = fopen(csvname, 'w');
            readData;
    end

    function readData
         if fidx >ftotal*2    
            fclose('all');
            text(80,100,'Done','FontSize',64,'Color', 'r','FontWeight','bold');
         else
            if fidx/2>floor(fidx/2)
                filen=filename((fidx+1)/2);
                filen=filen{1};
                if AnalyzeF
                    filen=filen(1:length(filen)-4);  % remove .hdr extension
                end
                DataTxt = uicontrol('Parent', TopPanel,'Position',[600,0,170,30], 'Style', 'text');
                set(DataTxt, 'BackgroundColor', [0.9,0.9,0.9],'FontSize', 14,'ForegroundColor', 'b','String', filen);
                fprintf(fid, '%s\n', filen); 
                fln = strcat(pathname, filen);
                
                if AnalyzeF
%                 oct=imread(fln);
                oct=uint8(readanalyze(fln));   % read Analyze image format
                oct=oct';
%                 oct=im2uint8(oct');  % convert to 8 bit image
                else
                     oct=imread(fln);
                     oct=im2uint8(oct);  % convert to 8 bit image
                     if size(oct,1)<1024
                        oct=cat(1, oct, zeros(1024-size(oct,1), size(oct,2)));  % paddle image with zeros if cropped.
                     end
                end
                    
                if EDIckF
%                     oct=flipud(im2uint8(oct));
                    oct=flipud(oct);
                else
%                     oct=[oct(1:10,:); oct(551:1024,:); oct(11:550,:)];  % move image
                    oct=[oct(1:10,:); oct(501:1024,:); oct(11:500,:)];
                end
                                
            end
            Data=oct(:, 51+mod(fidx-1,2)*700:250+mod(fidx-1,2)*700);
            
            if review
                idx=round(4.5*fidx-3.6);
                segdata=old_segdata(idx:idx+2,:)';
                segdata(segdata==-1)=nan;           % convert -1 back to Nan
            else
                segdata=Seg(Data);
            end
            drawline(Contrast,Data,segdata,scl);
         end
    end

    function EDIcheckCb(h, e)
        EDIckF = get(EDIcheck, 'value');
    end

%     function ImgPxcheckCb(h, e)
%         ImgPxF = get(ImgPxcheck, 'value');
%         ImgPixel=str2double(ImgPxTxt.String);
%     end

    function FoLDcheckCb(h, e)
        FoLDckF = get(FoLDcheck, 'value');
    end

    function CursorcheckCb(h, e)
         CursorckF = get(Cursorcheck, 'value');
         if CursorckF
            set(Cursorcheck, 'BackgroundColor', [1,1,1]);
             while 1           
                [x,y, butt] = ginput(1);
                if ~isequal(butt, 1)             % stop if not button 1
                    CursorckF=0;
                    set(Cursorcheck, 'BackgroundColor', [0.9,0.9,0.9]);
                    break
                end
                y=min(segdata(:,1))-11+round(y);
                if y<1; y=1; end
                if y>1024; y=1024; end
                [~,S]=min(abs(y-mean(segdata(:,1:3))));
                if x<1; x=1; end
                if x>200; x=200; end
                segdata(ceil(x/10),S)=y;
                drawline(Contrast,Data,segdata,scl);
            end
         end
    end

    function NGcheckCb(h, e)
         NGckF = get(NGcheck, 'value');
         if NGckF
            set(NGcheck, 'BackgroundColor', [1,1,1]);
             while 1           
                [x,y, butt] = ginput(1);
                if ~isequal(butt, 1)             % stop if not button 1
                    CursorckF=0;
                    set(NGcheck, 'BackgroundColor', [0.9,0.9,0.9]);
                    break
                end
                y=min(segdata(:,1))-11+round(y);
                if y<1; y=1; end
                if y>1024; y=1024; end
                [~,S]=min(abs(y-mean(segdata(:,1:3),'omitnan')))
                if x<1; x=1; end
                if x>200; x=200; end
                segdata(ceil(x/10),S)=NaN;
                drawline(Contrast,Data,segdata,scl);
            end
         end
    end

    function AllTracecheckCb(h, e)
         AllTraceckF = get(AllTracecheck, 'value');
    end

    function segChoiceCb(source,event)   % check for segmentation choice
        sgLayer = str2num(event.NewValue.String(1));
    end

    function CtsChoiceCb(source,event)   % check for Contrast choice
        Contrast = event.NewValue.String(1);
        drawline(Contrast,Data,segdata,scl);
    end

    function SlctSpCb(h,e)  % Select Strip no. 
        sno = round(get(h, 'Value'));
        DataTxt = uicontrol('Parent', TopPanel,'Position',[80,30,50,28], 'Style', 'text');
        set(DataTxt, 'BackgroundColor', [1,1,1],'FontSize', 14,'ForegroundColor', [0,0,0],'String', num2str(sno));
    end

%     function ScaleCb(h,e)  % Select scale for display OCT image 
%         scl = 1.15^(10-round(get(h, 'Value')));    % each step reduce 15%
%         drawline(Contrast,Data,segdata,scl);
%     end

    function UpCb(h,e) 
        if AllTraceckF
            segdata(:,sgLayer)=segdata(:,sgLayer)-1;
        else    
            segdata(sno,sgLayer)=segdata(sno,sgLayer)-1;
        end
        drawline(Contrast,Data,segdata,scl);
    end

    function DnCb(h,e) 
        if AllTraceckF
            segdata(:,sgLayer)=segdata(:,sgLayer)+1;
        else    
            segdata(sno,sgLayer)=segdata(sno,sgLayer)+1;
        end
        drawline(Contrast,Data,segdata,scl);
    end

    function checkend
        eyeidx = eyeidx+1;
        if eyeidx > length(eyefoldnames)
            cla;
            text(40,100,'Complete','FontSize',64,'Color', 'b','FontWeight','bold');  % end of all files
            return
        end
    end



    function AcceptCb(h,e) 
        if fidx >ftotal*2    
            if dayfoldf  
                checkend
                while isempty(dir([dayfold '\' eyefoldnames{eyeidx} '\*001.tiff']))
                    checkend
                end
                readfold([dayfold '\' eyefoldnames{eyeidx}])
             end
        else
            segdata(isnan(segdata))=-1;    % convert nan to -1 for .csv saving
            for j=1:3
                for i=1:20; fprintf(fid, '%g %s', segdata(i,j), ',');end
                fprintf(fid, '\n');
            end
            fprintf(fid, '\n');
            fidx=fidx+1;
            readData;
        end
    end

    function SeedcheckCb(h, e)
        [x,y] = ginput(1);
        y=min(segdata(:,1))-11+round(y);
        if y<1; y=1; end
        if y>1024; y=1024; end
%         [~,S]=min(abs(y-mean(segdata(:,1:3))));
        if x<1; x=1; end
        if x>200; x=200; end

        col=ceil(x/10);     % column marked for seed
        segdata(col,sgLayer)=y;
        switch sgLayer
            case 1
                
            case 2
                for i=col+1:20
                    pmax=findmaxima(smooth(mean(Data(segdata(i-1,2)-25:segdata(i-1,2)+25,i*10-9:i*10),2),5)); % mean intensity profile
                    [~, px]=min(abs(pmax-26));
                    segdata(i,2)=pmax(px)+segdata(i-1,2)-26;
                end
                for i=col-1:-1:1
                    pmax=findmaxima(smooth(mean(Data(segdata(i+1,2)-25:segdata(i+1,2)+25,i*10-9:i*10),2),5));
                    [~, px]=min(abs(pmax-26));
                    segdata(i,2)=pmax(px)+segdata(i+1,2)-26;
                end
            case 3
                for i=col+1:20
                    pfl=smooth(mean(Data(segdata(i-1,3)-25:segdata(i-1,3)+25,i*10-9:i*10),2),5); % mean intensity profile 20 points average
                    pmin=findminima(diff(pfl));
                    [~, px]=min(abs(pmin-25));
                    segdata(i,3)=pmin(px)+segdata(i-1,3)-24;
                end
                for i=col-1:-1:1
                    pfl=smooth(mean(Data(segdata(i+1,3)-25:segdata(i+1,3)+25,i*10-9:i*10),2),5); % mean intensity profile 20 points average
                    pmin=findminima(diff(pfl));
                    [~, px]=min(abs(pmin-25));
                    segdata(i,3)=pmin(px)+segdata(i+1,3)-24;
                end

        end
    
        drawline(Contrast,Data,segdata,scl);
    end

    function CSV_ClearCb(h, e)
        fclose('all');
        delete(csvname);
        cla
    end

   function AutocheckCb(h, e)
       segdata=Seg(Data);
       drawline(Contrast,Data,segdata,scl);     
   end

   function RAcheckCb(h, e)
    if FoLDckF
       fold_name=Foldpath;
    else
       fold_name = uigetdir;   % process fold list
    end

    files = dir([fold_name '\' '*_R_*.oct']);    % for all radial scan oct files
    pathname = [fold_name '\'];
    filename = {files.name};
    f = length(filename);

        
    h = waitbar(0,'Please wait...');
    
    NET.addAssembly('C:\Program Files\Bioptigen\ImageProcessingRecipes\RecipeUtilities.dll');
    Bioptigen.Recipes.Utility.RecipeManager.Instance.LoadRecipes('C:\ProgramData\Bioptigen\Recipes\MouseRecipe_Seg&Avg.xml');

    for i = 1:f
        fln = strcat(pathname, filename(i));
        Bioptigen.Recipes.Utility.RecipeManager.Instance.SetRecipeParameter('Input Filename', fln{1});

        Bioptigen.Recipes.Utility.RecipeManager.Instance.RunRecipe('Register and Average Processing');

        waitbar(i/f)
        
    end
        
    close(h)    
    InfoDisp('R&A Complete');    
%        RandA
   end

   function ExportCb(h, e)
    if FoLDckF
       fold_name=Foldpath;
    else
       fold_name = uigetdir;   
    end
    files = dir([fold_name '\' '*_RegAvg.oct']);   % for all registred and averaged oct files
    pathname = [fold_name '\'];
    filename = {files.name};
    f = length(filename);

    h = waitbar(0,'Please wait...');
    
    for i = 1:f
        extractOctFile(pathname,filename{i},'.tiff'); 
        
    end
        
    close(h)    
       
      InfoDisp('.tiff Export Complete'); 
%        tiffExport
   end

   function CombineCb(h, e)
       CombineCSV
   end

   function CombineTiffCb(h, e)
       Combine(dayfold)
       InfoDisp('Analyze Files Compiled');
   end


   function RegExpCb(h, e)
       RegExp
       InfoDisp('Reg Export Complete');
   end
    
   function RegCb(h, e)
      dos('Rscript Reg.R'); 
      InfoDisp('Registraton Complete!'); 
   end

   function IntensityCb(h, e)
      Intensity; 
      InfoDisp('Intensity Piled!'); 
   end

   function HypoCb(h, e)
      hypoband; 
      InfoDisp('Hypoband Calculated'); 
   end

   function InfoDisp(ITxt)
      InfoTxt = uicontrol('Parent', InfoPanel,'Position',[5,100,150,50], 'Style', 'text');
      set(InfoTxt,'BackgroundColor', [0.9,0.9,0.9],'FontSize', 14,'ForegroundColor', 'r','String',ITxt); 
   end

end


function segdata=Seg(x)
    Ioct=zeros(20,300);
    b=zeros(20,180);
    segdata=zeros(20,4);
    [~,idx]=max(smooth(mean(x,2),250));  % find starting of OCT band
    start=idx-149;
    if start>700 || start<0; start=700; end
    a=x(start:start+299,:);

    for i=1:4
        m=mean(a(:,i*50-49:i*50),2);     % averaged of 50 pixel
        k=sort(m, 'descend');
        t=(mean(k(1:100))-mean(k(201:300)))*0.2+mean(k(201:300));  % 20% of mean intensity start nfl
        idx=find(m>t);
        while idx(10)>idx(1)+10
            idx=idx(2:end);
        end
        
        for j=1:5
            Ioct((i-1)*5+j,:)=mean(a(:,(i-1)*50+j*10-9:(i-1)*50+j*10),2);     % averaged oct intensity 10 pixel
%             
            doct=diff(smooth(Ioct((i-1)*5+j,:),5));
            if idx(1)<10; idx(1)=11; end
%             if idx(1)>900; idx(1)=900; end
            [~,ilmx]=max(doct(idx(1)-10:idx(1)+10));
            segdata((i-1)*5+j,1)=ilmx+idx(1)-10;                   % ilm line
        end
    end
    
    segdata(segdata>121)=121;
    for i=1:20; b(i,:)=Ioct(i, segdata(i,1):segdata(i,1)+179); end
    
    c=mean(b);
    Soct20=smooth(c,20);
    Sub=smooth((c-Soct20'),5);
    SubMax=findmaxima(Sub);
    
    SubMax=SubMax(SubMax > 95 & SubMax <160);
    if length(SubMax)>4 
        [~,Maxi]=sort(Sub(SubMax),'descend');
        Maxi=Maxi(1:4); 
        Maxi=sort(Maxi);
        SubMax=SubMax(Maxi);
    end
    
    olm=SubMax(1);
    Submin=findminima(Sub);
    [~, bmx]=min(abs(Submin-145));
    bm=Submin(bmx);
    if bm-olm>38; bm=olm+38; end

    
    
    for i=1:20
        Soct=smooth(Ioct(i,:),5);
        Soct20=smooth(Ioct(i,:),20);
        Sub=smooth((Soct-Soct20'),5);
        SubMax=findmaxima(Sub);
        olmt=olm+segdata(i,1);
        
        [~, olmx]=min(abs(SubMax-olmt));
        
        segdata(i,2)=SubMax(olmx);
        
        bmt=segdata(i,2)- olm + bm;
        [~, bmx]=min(Soct(bmt-2:bmt+2));

        dmin=findminima(diff(Soct));
        segdata(i,3)=dmin(find(dmin<bmt + bmx+1,1,'last'))+1;


    end
        
    
    segdata=segdata+start-1;
end

function b=drawline(f,x, segment,scl)
    switch f
        case 'A'
            a=double(min(x(:)));
            b=double(max(x(:)));
            x=(x-a)*(255/(b-a));
        case 'S'
            a=double(min(x(:))*1.2);
            b=double(max(x(:))*0.9);
            x=(x-a)*(255/(b-a));
    end
    b=cat(3,x,x,x);
    for i=1:20
        for j=1:3
            if ~isnan(segment(i,j))
                for k=1:3
                    b(segment(i,j),(i-1)*10+1:i*10,k)=255*floor(sqrt(5-((k/j)-0.9)^2)/2.18);
                end
            end
        end
    end
    octlength = max(segment(:,3))-min(segment(:,1))+20;
    midline = min(segment(:,1))- 10 + octlength/2;
    lowline = midline - octlength*scl/2;
    if lowline < 1; lowline=1; end
    imshow(b(lowline:lowline+octlength*scl,:,:));
%     imshow(b(min(segment(:,1))-10:max(segment(:,3))+10,:,:))
end

function RandA
    if FoLDckF
       fold_name=Foldpath;
    else
       fold_name = uigetdir;   % process fold list
    end

    files = dir([fold_name '\' '*_R_*.oct']);    % for all radial scan oct files
    pathname = [fold_name '\'];
    filename = {files.name};
    f = length(filename);

        
    h = waitbar(0,'Please wait...');
    
    NET.addAssembly('C:\Program Files\Bioptigen\ImageProcessingRecipes\RecipeUtilities.dll');
    Bioptigen.Recipes.Utility.RecipeManager.Instance.LoadRecipes('C:\ProgramData\Bioptigen\Recipes\MouseRecipe_Seg&Avg.xml');

    for i = 1:f
        fln = strcat(pathname, filename(i));
        Bioptigen.Recipes.Utility.RecipeManager.Instance.SetRecipeParameter('Input Filename', fln{1});

        Bioptigen.Recipes.Utility.RecipeManager.Instance.RunRecipe('Register and Average Processing');

        waitbar(i/f)
        
    end
        
    close(h)    

end
 
function tiffExport
    if FoLDckF
       fold_name=Foldpath;
    else
       fold_name = uigetdir;   
    end
    files = dir([fold_name '\' '*_RegAvg.oct']);   % for all registred and averaged oct files
    pathname = [fold_name '\'];
    filename = {files.name};
    f = length(filename);

    h = waitbar(0,'Please wait...');
    
    for i = 1:f
        extractOctFile(pathname,filename{i},'.tiff'); 
        
    end
        
    close(h)    

end
 
function CombineCSV

    fold_name = uigetdir;

%     cd(fold_name);
    x = strfind(fold_name, '\');
    fid = fopen([fold_name '\' fold_name(x(end)+1:end) '_seg.csv'], 'w');
    fprintf(fid, '%s', 'FileName');

    files = dir(fold_name);   % process fold list
    filenames = {files.name};
    subdirs = filenames([files.isdir]);

    h = waitbar(0,'Please wait...');
    
    for s = 3:length(subdirs)
        if exist([fold_name '\' subdirs{s} '\' subdirs{s} '.csv'], 'file')
            fprintf(fid, '%s', [',' subdirs{s}]);
%             cd ([fold_name '\' subdirs{s}]);
            old_segdata = xlsread([fold_name '\' subdirs{s} '\' subdirs{s} '.csv']);            % Read data
            
            nof=(size(old_segdata,1)+2)*2/9;      % number of frames *2
%             for fidx=1:8
            for fidx=1:nof
                idx=round(4.5*fidx-3.6);
                segdata=old_segdata(idx:idx+2,:);
                segdata(segdata==-1)=nan;           % convert -1 back to Nan
                
                OutRetina(s-2,fidx) =1.53*(mean(segdata(3,:),'omitnan')-mean(segdata(2,:),'omitnan'));   % outer retina thickness
                TotalRetina(s-2,fidx)=1.53*(mean(segdata(3,:),'omitnan')-mean(segdata(1,:),'omitnan'));   % total retina thickness
                OLM(s-2,fidx)=mean(segdata(2,:),'omitnan');
            end
        end
        waitbar(s / length(subdirs))
    end

    close(h) 
    fprintf(fid, '\n');

%     cd(fold_name);


%     for i=1:8
    for i=1:nof
        fprintf(fid, '%s','Outer Retina');
        for s=3:length(subdirs)
            if exist([fold_name '\' subdirs{s} '\' subdirs{s} '.csv'], 'file')
                fprintf(fid, '%s',',');
                fprintf(fid, '%8.2f',OutRetina(s-2,i));
            end
        end
        fprintf(fid, '\n');

    end
    fprintf(fid, '\n%s','AVG');
        for s=3:length(subdirs)
            if exist([fold_name '\' subdirs{s} '\' subdirs{s} '.csv'], 'file')
                fprintf(fid, '%s',',');
                fprintf(fid, '%8.2f',mean(OutRetina(s-2,:)));
            end
        end
        fprintf(fid, '\n');
    
    fprintf(fid, '%s','Diff');
        for s=3:length(subdirs)
            if exist([fold_name '\' subdirs{s} '\' subdirs{s} '.csv'], 'file')
                fprintf(fid, '%s',',');
                fprintf(fid, '%8.2f',max(OutRetina(s-2,:))-min(OutRetina(s-2,:)));
            end
        end
        fprintf(fid, '\n\n');
%     fprintf(fid, '\n')
%     for i=1:3; fprintf(fid, '\n'); end

%     for i=1:8
    for i=1:nof
        fprintf(fid, '%s','Total Retina');
        for s=3:length(subdirs)
            if exist([fold_name '\' subdirs{s} '\' subdirs{s} '.csv'], 'file')
                fprintf(fid, '%s',',');
                fprintf(fid, '%8.2f',TotalRetina(s-2,i));
            end
        end
        fprintf(fid, '\n');

    end

    for i=1:3; fprintf(fid, '\n'); end

%     for i=1:8
    for i=1:nof
        fprintf(fid, '%s','OLM line');
        for s=3:length(subdirs)
            if exist([fold_name '\' subdirs{s} '\' subdirs{s} '.csv'], 'file')
                fprintf(fid, '%s',',');
                fprintf(fid, '%8.2f',OLM(s-2,i));
            end
        end
        fprintf(fid, '\n');

    end

    fclose('all');

end

