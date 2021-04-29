%  Draw segline on OCT image based on OCTseg results
%  2021-07-29, modified from Intensity.m


% function Intensity
    eyefoldnames=[];  % list of fold names for OCT data
%     eyeidx=1;   % eye index
    dayfold=[];   % path for day fold
%     dayfoldf=0;  % flag for read more than one eye data
% 
    files = {};  % file cell array
    filename = {};  % filename cell array
    pathname = [];  % pathname
%     ftotal=1; %total image files
%     segdata=[];  % segmented data
%     fidx=1;  %image index
    fln=[];  % image filename

    DayfoldCb
    




    function DayfoldCb(h,e)  % load day fold
        dayfold = uigetdir;   % process fold list
        eyefold = dir(dayfold);
%         eyeidx=1;
%         dayfoldf=1;
        eyefoldnames = {eyefold.name};
        dirFlags = [eyefold.isdir] & ~strcmp(eyefoldnames, '.') & ~strcmp(eyefoldnames, '..');
        eyefoldnames = eyefoldnames(dirFlags);
        
        k=strfind(dayfold, '\');
        k=k(end);
%         fid = fopen(strcat(dayfold, dayfold(k:end),'_intensity.csv'), 'w');
%         fdx=0;
        h = waitbar(0,'Please wait...');
        numfile=length(eyefoldnames);

        for eyeidx=1:numfile
            while isempty(dir([dayfold '\' eyefoldnames{eyeidx} '\*001.tiff*']))
                eyeidx=eyeidx+1;
            end
            drawimage([dayfold '\' eyefoldnames{eyeidx}]);
            
            waitbar(eyeidx/length(eyefoldnames))
        end
        

        close(h)
    end

    function drawimage(fold_name)
         files = dir([fold_name '\' '*.tiff']);   
         pathname = [fold_name '\'];
         filename = {files.name};
%          ftotal = length(filename);
         
         k=strfind(fold_name, '\');
         k=k(end);
         csvname=strcat(pathname, fold_name(k+1:end), '.csv');
         old_segdata=xlsread(csvname);
       
%          AVG=[];     % averaged intensity
%          ORT=[];    % outer retina thickness in pixel (OLM to RPE)
         for fidx=1:4  % number of frames
             filen=filename(fidx);
             filen=filen{1};
             fln = strcat(pathname, filen);
             oct=imread(fln);
%              if size(oct,1)<1024
%                  oct=cat(1, oct, zeros(1024-size(oct,1), size(oct,2)));  % paddle image with zeros if cropped.
%              end
             oct=flipud(im2uint8(oct));  % convert to 8 bit image
             
             coloroct=cat(3, oct, oct, oct);
             
             
     for i=1:20
        for j=1:3
            y1=old_segdata(round(9*fidx-7)-2+j,i);
            y2=old_segdata(round(9*fidx-7)+2+j,i);
%             if ~isnan(y)
                
                for k=1:3
                    coloroct(y1,(i-1)*10+51:i*10+50,k)=255*floor(sqrt(5-((k/j)-0.9)^2)/2.18);
                    coloroct(y2,(i-1)*10+751:i*10+750,k)=255*floor(sqrt(5-((k/j)-0.9)^2)/2.18);
                end
%             end
        end
    end

             
             imwrite(coloroct,[fln(1:end-5) '_seg.tiff']);
             
             
             
         end
         
    end
        
